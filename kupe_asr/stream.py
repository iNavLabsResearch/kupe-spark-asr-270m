"""Streaming inference with AssemblyAI-style turn flags.

Two flags, both derived from the model's own end-of-sequence probability, so no
extra training or model is needed:

  PRE_HIT_LLM     P(eos) crosses `pre_llm_threshold` while still decoding.
                  -> the draft transcript is stable enough to *prefetch* your LLM.
  END_OF_SPEECH   eos actually emitted, or P(eos) >= `eos_threshold`, or a VAD
                  silence tail -> the turn is over; commit the final transcript.

Language: pass "auto" to let the model emit the detected <|lang_xx|> as its first
token (exposed as `language` in events); or pass a code ("hi","en",...) to force it.

Note: each hop re-encodes the bounded audio buffer and re-decodes greedily. Correct
and simple; swap in KV-cache incremental decoding for lowest latency at scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .constants import LANG_CODES, MIMI_SAMPLE_RATE
from .tokenizer import load_tokenizer, token_map

PARTIAL = "partial"
PRE_HIT_LLM = "pre_hit_llm"
END_OF_SPEECH = "end_of_speech"


@dataclass
class Event:
    type: str
    text: str
    language: str | None
    t: float          # seconds of audio consumed so far


class StreamingASR:
    def __init__(self, model_dir: str, mimi_id: str = "kyutai/mimi",
                 device: str | None = None, *, pre_llm_threshold: float = 0.30,
                 eos_threshold: float = 0.85, max_context_s: float = 30.0,
                 silence_ms: float = 800.0, silence_rms: float = 0.01,
                 max_new_tokens: int = 200):
        from transformers import AutoFeatureExtractor, AutoModelForCausalLM, MimiModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = load_tokenizer(model_dir)
        self.m = token_map(self.tok)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir).to(self.device).eval()
        self.fe = AutoFeatureExtractor.from_pretrained(mimi_id)
        self.mimi = MimiModel.from_pretrained(mimi_id).to(self.device).eval()

        self.pre_llm_threshold = pre_llm_threshold
        self.eos_threshold = eos_threshold
        self.max_context_s = max_context_s
        self.silence_ms = silence_ms
        self.silence_rms = silence_rms
        self.max_new_tokens = max_new_tokens
        self.reset()

    def reset(self) -> None:
        self.buffer = np.zeros(0, dtype=np.float32)
        self._fired_pre = False
        self._done = False

    # --------------------------------------------------------------- input
    def add_audio(self, chunk: np.ndarray, sr: int) -> None:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if sr != MIMI_SAMPLE_RATE:
            import librosa
            chunk = librosa.resample(chunk, orig_sr=sr, target_sr=MIMI_SAMPLE_RATE)
        self.buffer = np.concatenate([self.buffer, chunk])
        # bound context
        max_len = int(self.max_context_s * MIMI_SAMPLE_RATE)
        if len(self.buffer) > max_len:
            self.buffer = self.buffer[-max_len:]

    # --------------------------------------------------------------- codec
    @torch.inference_mode()
    def _encode(self) -> list[int]:
        inputs = self.fe(raw_audio=self.buffer, sampling_rate=MIMI_SAMPLE_RATE,
                         return_tensors="pt")
        iv = inputs["input_values"].to(self.device)
        pm = inputs.get("padding_mask")
        pm = pm.to(self.device) if pm is not None else None
        out = self.mimi.encode(iv, pm, num_quantizers=1)
        return out.audio_codes[0, 0, :].tolist()

    def _trailing_silence(self) -> bool:
        n = int(self.silence_ms / 1000.0 * MIMI_SAMPLE_RATE)
        if len(self.buffer) < n:
            return False
        tail = self.buffer[-n:]
        return float(np.sqrt(np.mean(tail ** 2) + 1e-9)) < self.silence_rms

    # --------------------------------------------------------------- decode
    @torch.inference_mode()
    def step(self, lang: str = "auto") -> list[Event]:
        if self._done:
            return []
        t = len(self.buffer) / MIMI_SAMPLE_RATE
        codes = self._encode()
        audio_ids = [self.m.mimi_base + int(c) for c in codes]
        control = self.m.lang_auto if lang == "auto" else self.m.lang[lang]
        prefix = [self.m.bos, self.m.audio_start] + audio_ids + [self.m.audio_end, control]
        input_ids = torch.tensor([prefix], device=self.device)

        gen = self.model.generate(
            input_ids=input_ids, max_new_tokens=self.max_new_tokens,
            do_sample=False, num_beams=1, eos_token_id=self.m.eos,
            pad_token_id=self.m.pad, output_scores=True, return_dict_in_generate=True,
        )
        new_ids = gen.sequences[0, len(prefix):].tolist()

        # detected language (auto mode) is the first generated token
        detected = lang
        text_ids = new_ids
        if lang == "auto" and new_ids and new_ids[0] in self.m.lang_id_to_code:
            detected = self.m.lang_id_to_code[new_ids[0]]
            text_ids = new_ids[1:]

        # per-step P(eos) to drive the flags
        pre_step = None
        eos_hit = False
        for i, logits in enumerate(gen.scores):
            p_eos = torch.softmax(logits[0].float(), dim=-1)[self.m.eos].item()
            if pre_step is None and p_eos >= self.pre_llm_threshold:
                pre_step = i
            if p_eos >= self.eos_threshold or (i < len(new_ids) and new_ids[i] == self.m.eos):
                eos_hit = True
                break

        # strip eos from text
        if self.m.eos in text_ids:
            text_ids = text_ids[: text_ids.index(self.m.eos)]
        text = self.tok.decode(text_ids, skip_special_tokens=True).strip()

        events = [Event(PARTIAL, text, detected, t)]

        if pre_step is not None and not self._fired_pre:
            self._fired_pre = True
            draft_ids = new_ids[: pre_step]
            if lang == "auto" and draft_ids and draft_ids[0] in self.m.lang_id_to_code:
                draft_ids = draft_ids[1:]
            draft = self.tok.decode(draft_ids, skip_special_tokens=True).strip()
            events.append(Event(PRE_HIT_LLM, draft or text, detected, t))

        if eos_hit or self._trailing_silence():
            self._done = True
            events.append(Event(END_OF_SPEECH, text, detected, t))

        return events


def transcribe_file(model_dir: str, wav_path: str, lang: str = "auto",
                    chunk_ms: float = 320, **kw):
    """Simulate streaming over a wav file; yield Events in order."""
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    asr = StreamingASR(model_dir, **kw)
    hop = int(sr * chunk_ms / 1000.0)
    for start in range(0, len(audio), hop):
        asr.add_audio(audio[start : start + hop], sr)
        for ev in asr.step(lang=lang):
            yield ev
            if ev.type == END_OF_SPEECH:
                return
