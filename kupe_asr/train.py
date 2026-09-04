"""Training entrypoint: HF Trainer over the Mimi dataset, with a live WER callback,
and an end-of-run full eval whose report + weights are pushed to the runs repo.

Multi-GPU: launch with `accelerate launch` or `torchrun` — Trainer handles DDP.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import time

import torch
from transformers import TrainerCallback

from .data.collate import AsrCollator, SequenceBuilder
from .data.load import load_mimi_dataset, splits_from_dataset
from .evaluate import run_eval, save_report
from .hf_utils import ensure_repo, hf_login, hf_token, init_wandb, log, upload_folder
from .model import load_model
from .tokenizer import build_tokenizer, load_tokenizer, token_map


class HubCheckpointCallback(TrainerCallback):
    """Push the latest checkpoint (weights + optimizer) to runs/<run>/last_checkpoint
    each save, so any box can resume/extend the run from the Hub."""

    def __init__(self, cfg, run_name):
        self.cfg = cfg
        self.run_name = run_name

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt):
            return control
        try:
            ensure_repo(self.cfg.repos.runs, "model")
            upload_folder(ckpt, self.cfg.repos.runs, "model",
                          path_in_repo=f"runs/{self.run_name}/last_checkpoint",
                          commit_message=f"{self.run_name}: checkpoint @ step {state.global_step}")
            log.info("pushed checkpoint step %d to Hub (resumable)", state.global_step)
        except Exception as e:
            log.warning("checkpoint push failed: %s", e)
        return control


def _resume_checkpoint(cfg, run_name, hf_dir):
    """Return a checkpoint path to resume from: newest local, else pulled from Hub."""
    local = sorted(glob.glob(os.path.join(hf_dir, "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1)
    if local:
        log.info("resuming from local checkpoint %s", local[-1])
        return local[-1]
    try:
        from huggingface_hub import snapshot_download
        p = snapshot_download(cfg.repos.runs, repo_type="model", token=hf_token(),
                              allow_patterns=[f"runs/{run_name}/last_checkpoint/*"])
        src = os.path.join(p, "runs", run_name, "last_checkpoint")
        if os.path.isdir(src) and os.listdir(src):
            dest = os.path.join(hf_dir, "checkpoint-hub")
            os.makedirs(hf_dir, exist_ok=True)
            shutil.copytree(src, dest, dirs_exist_ok=True)
            log.info("resuming from Hub checkpoint runs/%s/last_checkpoint", run_name)
            return dest
    except Exception as e:
        log.warning("no Hub checkpoint for run %s (%s) — starting fresh", run_name, e)
    return None


def _tokenizer_exists(d: str) -> bool:
    return os.path.exists(os.path.join(d, "tokenizer_config.json"))


class WerCallback(TrainerCallback):
    """Lightweight generation-based WER on a small subset at every eval step."""

    def __init__(self, model, tokenizer, tmap, val_ds, cfg):
        self.model = model
        self.tok = tokenizer
        self.tmap = tmap
        self.cfg = cfg
        # fixed small subset for speed/repeatability
        n = min(val_ds.num_rows, 6 * 40)
        self.subset = val_ds.shuffle(seed=0).select(range(n))

    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:   # rank 0 only under DDP
            return control
        try:
            rep = run_eval(
                self.model, self.tok, self.tmap, self.subset, self.model.device,
                max_audio_frames=self.cfg.model.max_audio_frames,
                max_samples_per_lang=40, auto_samples_per_lang=20,
                max_new_tokens=self.cfg.eval.max_new_tokens,
            )
            metrics = {
                "eval/wer": rep["overall"]["wer"],
                "eval/cer": rep["overall"]["cer"],
                "eval/auto_lang_acc": rep["auto"]["lang_id_acc"],
            }
            log.info("step %d | WER=%.3f CER=%.3f langID=%.3f",
                     state.global_step, metrics["eval/wer"],
                     metrics["eval/cer"], metrics["eval/auto_lang_acc"])
            if not os.environ.get("WANDB_DISABLED"):
                import wandb
                if wandb.run is not None:
                    wandb.log(metrics, step=state.global_step)
        except Exception as e:  # never crash training on eval
            log.warning("WER callback failed: %s", e)
        return control


def _training_args(cfg, out_dir, run_name, use_wandb, *, overwrite: bool):
    from transformers import TrainingArguments

    def _opt(**kw):
        """Build TrainingArguments, dropping kwargs unknown to this transformers version."""
        import inspect
        allowed = set(inspect.signature(TrainingArguments.__init__).parameters)
        return TrainingArguments(**{k: v for k, v in kw.items() if k in allowed})

    return _opt(
        output_dir=os.path.join(out_dir, "hf"),
        overwrite_output_dir=overwrite,
        num_train_epochs=cfg.train.epochs,
        max_steps=int(getattr(cfg.train, "max_steps", 0) or -1),
        per_device_train_batch_size=cfg.train.per_device_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=float(cfg.train.lr),
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=bool(cfg.train.bf16),
        tf32=True,                          # Ampere+ matmul speedup (H100/4090/PRO6000)
        optim="adamw_torch_fused",          # fused AdamW -> faster, less overhead
        gradient_checkpointing=bool(cfg.train.gradient_checkpointing),
        group_by_length=True,               # batch similar lengths -> less padding, full GPU
        length_column_name="num_frames",
        logging_steps=cfg.train.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.train.eval_steps,
        prediction_loss_only=True,          # never gather 258k-vocab logits (OOM); WER via callback
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        save_safetensors=True,
        dataloader_num_workers=cfg.train.num_workers,
        dataloader_pin_memory=True,
        report_to=["wandb"] if use_wandb else ["none"],
        run_name=run_name,
        remove_unused_columns=False,        # keep mimi_cb0/text/language for the collator
        ddp_find_unused_parameters=False,
        seed=cfg.seed,
        logging_first_step=True,
    )


def train(cfg, *, from_hub: bool = False, resume: str | None = None) -> str:
    from transformers import Trainer, set_seed

    set_seed(cfg.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation
    torch.backends.cuda.matmul.allow_tf32 = True     # full-throughput matmul on Ampere+
    torch.backends.cudnn.allow_tf32 = True
    hf_login()

    if resume:                                        # extend an existing run
        run_name = resume
        overwrite = False
        log.info("=== extending run %s ===", run_name)
    else:
        run_name = f"{cfg.project}-{time.strftime('%Y%m%d-%H%M%S')}"
        overwrite = True
    out_dir = os.path.join(cfg.paths.runs_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    use_wandb = init_wandb(cfg.project, run_name, cfg.to_dict()) is not None

    # tokenizer (build once, reuse)
    tdir = cfg.paths.tokenizer_dir
    tok = load_tokenizer(tdir) if _tokenizer_exists(tdir) else build_tokenizer(cfg.model.base_id, tdir)
    tmap = token_map(tok)

    # model
    model = load_model(cfg.model.base_id, tmap.vocab_size, cfg.model.dtype, cfg.model.attn_impl)

    # data: local mimi dataset if present, otherwise Hub `mimi` config
    train_ds, val_ds = splits_from_dataset(load_mimi_dataset(cfg, from_hub=from_hub))
    log.info("train=%d val=%d", train_ds.num_rows, val_ds.num_rows)

    builder = SequenceBuilder(tok, tmap, cfg.model.max_seq_len,
                              cfg.model.max_audio_frames, cfg.model.max_text_tokens)
    collator = AsrCollator(builder, tmap, cfg.train.p_auto, cfg.seed)

    args = _training_args(cfg, out_dir, run_name, use_wandb, overwrite=overwrite)
    callbacks = [WerCallback(model, tok, tmap, val_ds, cfg)]
    if bool(getattr(cfg.train, "push_checkpoints", True)):
        callbacks.append(HubCheckpointCallback(cfg, run_name))
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, processing_class=tok,
        callbacks=callbacks,
    )

    resume_ckpt = _resume_checkpoint(cfg, run_name, args.output_dir) if resume else None
    log.info("=== training %s (resume=%s) ===", run_name, bool(resume_ckpt))
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # only the main process saves / evaluates / pushes (DDP-safe)
    if hasattr(trainer, "accelerator"):
        trainer.accelerator.wait_for_everyone()
    if not trainer.is_world_process_zero():
        log.info("rank %s done (no push)", args.process_index)
        return out_dir

    # save final model + tokenizer
    model_dir = os.path.join(out_dir, "model")
    trainer.save_model(model_dir)
    tok.save_pretrained(model_dir)

    # dump run config + trainer history
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    with open(os.path.join(out_dir, "trainer_state.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    # === full end-of-run eval ===
    log.info("=== final evaluation ===")
    device = model.device
    report = run_eval(
        model, tok, tmap, val_ds, device,
        max_audio_frames=cfg.model.max_audio_frames,
        max_samples_per_lang=cfg.eval.max_samples_per_lang,
        auto_samples_per_lang=cfg.eval.auto_samples_per_lang,
        max_new_tokens=cfg.eval.max_new_tokens,
    )
    save_report(report, out_dir, title=run_name)
    log.info("FINAL WER=%.3f CER=%.3f langID=%.3f",
             report["overall"]["wer"], report["overall"]["cer"],
             report["auto"]["lang_id_acc"])

    # === push the whole run (weights + reports) to the runs repo ===
    if cfg.train.push_to_hub:
        ensure_repo(cfg.repos.runs, "model")
        upload_folder(out_dir, cfg.repos.runs, "model",
                      path_in_repo=f"runs/{run_name}",
                      commit_message=f"run {run_name}: WER={report['overall']['wer']:.3f}",
                      ignore_patterns=["hf/**", "hf/checkpoint-*/**"])  # skip raw checkpoints
        # also refresh the canonical model repo with the latest weights
        ensure_repo(cfg.repos.model, "model")
        upload_folder(model_dir, cfg.repos.model, "model",
                      commit_message=f"latest: {run_name} WER={report['overall']['wer']:.3f}")

    log.info("done. run dir: %s", out_dir)
    return out_dir
