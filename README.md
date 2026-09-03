# kupe-spark-asr-270m

Streaming **multilingual ASR** on **Gemma-3-270m + Mimi** codec.
Audio → Mimi codebook-0 tokens (12.5 tok/s, semantic) → Gemma decodes text.

- **Languages:** `en`, `hi`, `gu`, `bn`, `ur`, `mr` (English + Devanagari Hindi/Marathi, Gujarati, Bengali, Urdu).
- **Auto language:** pass `auto` and the model emits the detected `<|lang_xx|>` as its first token; pass a code (`hi`) to force it.
- **Streaming turn flags** (AssemblyAI-style): `PRE_HIT_LLM` (prefetch your LLM) and `END_OF_SPEECH` (commit final).
- **One training run.** Encode once → train → auto-eval → auto-push weights + eval report to the runs repo.

Everything is driven by [`configs/config.yaml`](configs/config.yaml).

---

## 0. Prerequisites (do these once)

**Accept the licenses/gates** (same HF account as your token):
- Model: <https://huggingface.co/google/gemma-3-270m>
- Data: <https://huggingface.co/datasets/ai4bharat/IndicVoices> ·
  <https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part> ·
  <https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0>

**Secrets** — copy `.env.example` → `.env` and fill it in, *or* export:

```bash
export HF_TOKEN=hf_xxx
export HF_OWNER=anuj-inavlabs
export WANDB_API_KEY=xxx
export WANDB_PROJECT=kupe-spark-asr-270m
```

**Install** (install torch matched to your CUDA first):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

## 1. Run the whole pipeline

**Two machines (recommended):** CPU VM fetches and pushes `audio`; GPU VM pulls it, encodes, pushes `mimi`, trains.

```bash
# --- CPU VM (no GPU needed; ~300 GB disk, use --hub-only so local shards are deleted after push) ---
python scripts/00_create_repos.py
python scripts/01_fetch_data.py --hub-only     # stream → 24 kHz → push `audio` → delete local

# --- GPU VM (empty artifacts/ is fine; scripts pull from Hub if local is missing) ---
python scripts/02_encode_data.py --from-hub    # download `audio`, Mimi-encode, push `mimi`
python scripts/03_train.py                     # local `mimi` if 02 just ran; else --from-hub
python scripts/04_evaluate.py --push
```

Same box (needs enough disk for a local audio copy as well as the Hub push):

```bash
python scripts/00_create_repos.py
python scripts/01_fetch_data.py       # stream, resample→24kHz, keep local + push `audio`
python scripts/02_encode_data.py      # Mimi-encode → push `mimi` config (this is what we train on)
python scripts/03_train.py            # train + live WER + final eval + push run to the runs repo
python scripts/04_evaluate.py --push  # (re)evaluate a checkpoint, push the report
```

Multi-GPU training:

```bash
accelerate launch scripts/03_train.py
```

Streaming demo (prints the two flags):

```bash
python scripts/05_stream_demo.py --model-dir artifacts/runs/<run>/model --wav sample.wav --lang auto
```

Or use the Makefile: `make repos && make fetch && make encode && make train && make eval`.

---

## 2. What the streaming output looks like

```
[  1.28s] …            [hi]  नमस्ते मैं
[  2.24s] …            [hi]  नमस्ते मैं आपकी कैसे
[  2.56s] ⚡PRE_HIT_LLM [hi]  नमस्ते मैं आपकी कैसे मदद        ← prefetch your LLM now
[  3.20s] ■END_OF_SPEECH[hi] नमस्ते मैं आपकी कैसे मदद कर सकता हूँ  ← commit final
```

Both flags come from the model's own `P(<eos>)` (thresholds in `configs/config.yaml`):
`pre_llm_threshold` fires `PRE_HIT_LLM`; `eos_threshold` / real eos / a silence tail fires `END_OF_SPEECH`. No second model, no extra training.

---

## 3. Auto language detection

Training mixes two modes per clip (`train.p_auto`, default 0.5):

```
forced : <bos><audio_start> …audio… <audio_end> <lang_hi>  <text> <eos>
auto   : <bos><audio_start> …audio… <audio_end> <lang_auto> <lang_hi> <text> <eos>
```

So at inference:
- `--lang auto` → first generated token **is** the detected language (`event.language`).
- `--lang hi` → language is forced; the model transcribes Hindi directly.

---

## 4. Data & hours

Streaming with per-language caps means **we download only what we keep**. Defaults in `configs/config.yaml`:

| lang | target cap | realistic kept | main sources |
|---|---|---|---|
| en | 300 h | ~300 h | Common Voice 17 |
| hi | 400 h | ~400 h | Vaani (963h avail) + IndicVoices + CV |
| bn | 200 h | ~200 h | Vaani (154h) + IndicVoices + CV |
| mr | 120 h | ~120 h | Vaani (68h) + IndicVoices + CV |
| gu | 60 h  | ~40–60 h | Vaani (19h) + IndicVoices + CV |
| ur | 40 h  | ~30–40 h | Vaani (12h) + IndicVoices + CV |
| **total** | **~1,120 h** | **~1,050–1,120 h** | |

Raise/lower `data.max_hours_per_lang` to taste. At 12.5 tok/s, ~1,100 h ≈ **~50 M audio tokens/epoch** — tiny.

---

## 5. Training time (rough, single GPU)

Compute for ~1,100 h, codebook-0 only, 3 epochs (`C≈6·N·D`, N=270M, D≈0.2B tok). Wall-clock ≈ compute × ~1.5–2.

| GPU | encode (once) | train (3 ep) |
|---|---|---|
| **H100 80GB** | ~2–4 h | **~0.5–1 h** |
| **RTX PRO 6000 (Blackwell 96GB)** | ~3–5 h | ~1–1.5 h |
| **RTX 4090 24GB** | ~5–8 h | ~2–3 h |
| **L4 24GB** | ~8–12 h | ~3–5 h |

The **encode + download** dominate, not training. All GPUs above fit full fine-tuning of 270M easily (raise `mimi.batch_max_frames` and `train.per_device_batch_size` on big cards).

---

## 6. Repos (on `anuj-inavlabs`)

- `kupe-spark-asr-270m-data` — dataset, two configs: **audio** (raw) + **mimi** (tokens). Viewer works for both.
- `kupe-spark-asr-270m-runs` — every run pushed to `runs/<run>/` (weights + `eval_report.md`/`.json` + configs + trainer history).
- `kupe-spark-asr-270m` — latest model weights.

## 7. Layout

```
configs/config.yaml     one config for all stages
kupe_asr/
  constants.py          languages, special tokens, codec geometry
  config.py hf_utils.py tokenizer.py model.py text.py
  data/ sources.py fetch.py encode.py shards.py collate.py
  train.py evaluate.py stream.py
scripts/ 00..05         one file per stage
```

## 8. Notes

- **Semantic-only (codebook 0)** keeps input at 12.5 tok/s — the efficient ASR path. To trade sequence length for acoustic detail, raise `mimi.num_codebooks` (needs a flattening tweak in the collator).
- Sources are tried independently; a gated/missing config is **logged and skipped**, never fatal.
- Column names are auto-detected (IndicVoices/Vaani don't document them), so schema drift won't crash the loader.
- If `WANDB_API_KEY` is unset, W&B is disabled and metrics still print.
```
