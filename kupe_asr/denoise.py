"""Streaming RNNoise: clean mic audio before Mimi/Gemma.

RNNoise is 48 kHz / 10 ms frames. We resample with soxr, denoise, then
resample to Mimi's 24 kHz. Incomplete frames stay buffered across chunks.
"""
from __future__ import annotations

import numpy as np

from .constants import MIMI_SAMPLE_RATE
from .hf_utils import log

RNNOISE_SR = 48_000
RNNOISE_FRAME = 480  # 10 ms at 48 kHz


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if sr_in == sr_out or x.size == 0:
        return x
    try:
        import soxr
        return np.asarray(soxr.resample(x, sr_in, sr_out), dtype=np.float32)
    except Exception:
        import librosa
        return np.asarray(librosa.resample(x, orig_sr=sr_in, target_sr=sr_out), dtype=np.float32)


def _to_i16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float32) * 32767.0, -32768, 32767).astype(np.int16)


def _to_f32(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x).reshape(-1)
    if arr.dtype == np.int16:
        return arr.astype(np.float32) / 32768.0
    return np.asarray(arr, dtype=np.float32)


class StreamingRNNoise:
    """Stateful RNNoise for a live session. Safe no-op if pyrnnoise is missing."""

    def __init__(self):
        self._pending48 = np.zeros(0, dtype=np.float32)
        self._impl = None
        self._state = None
        self._create = None
        self._destroy = None
        self._process = None
        self._rn = None
        self._frame = RNNOISE_FRAME
        self._load()

    @property
    def enabled(self) -> bool:
        return self._impl is not None

    def _load(self) -> None:
        try:
            from pyrnnoise.rnnoise import FRAME_SIZE, create, destroy, process_frame
            self._create, self._destroy, self._process = create, destroy, process_frame
            self._frame = int(FRAME_SIZE)
            self._state = [create()]
            self._impl = "capi"
            log.info("RNNoise ready (pyrnnoise C API, %d-sample frames @ 48 kHz)", self._frame)
            return
        except Exception as e:
            capi_err = e
        try:
            from pyrnnoise import RNNoise
            self._rn = RNNoise(sample_rate=RNNOISE_SR)
            self._impl = "chunk"
            log.info("RNNoise ready (pyrnnoise denoise_chunk)")
            return
        except Exception as e:
            log.warning(
                "RNNoise unavailable (%s / %s) — installing: pip install pyrnnoise",
                capi_err, e,
            )
            self._impl = None

    def reset(self) -> None:
        self._pending48 = np.zeros(0, dtype=np.float32)
        if self._impl == "capi" and self._create is not None:
            if self._state:
                for s in self._state:
                    try:
                        self._destroy(s)
                    except Exception:
                        pass
            self._state = [self._create()]
        elif self._impl == "chunk" and self._rn is not None:
            try:
                self._rn.reset()
            except Exception:
                pass

    def _denoise_i16(self, i16: np.ndarray) -> np.ndarray:
        if self._impl == "capi":
            frame = np.ascontiguousarray(i16.reshape(1, -1))
            a, b = self._process(self._state, frame)
            aa, bb = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
            denoised = aa if aa.size >= bb.size else bb
            return denoised
        out = []
        for _prob, frame in self._rn.denoise_chunk(i16.reshape(1, -1), False):
            out.append(np.asarray(frame).reshape(-1))
        return np.concatenate(out) if out else i16

    def process(self, pcm: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """Clean `pcm` and return (audio, sample_rate) at Mimi 24 kHz.

        May return an empty array if we don't yet have a full 10 ms RNNoise frame.
        """
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if pcm.size == 0:
            return pcm, MIMI_SAMPLE_RATE
        if not self.enabled:
            return _resample(pcm, sr, MIMI_SAMPLE_RATE), MIMI_SAMPLE_RATE

        x48 = _resample(pcm, sr, RNNOISE_SR)
        self._pending48 = np.concatenate([self._pending48, x48])
        cleaned = []
        n = self._frame
        while len(self._pending48) >= n:
            frame = self._pending48[:n]
            self._pending48 = self._pending48[n:]
            denoised = self._denoise_i16(_to_i16(frame))
            cleaned.append(_to_f32(denoised))
        if not cleaned:
            return np.zeros(0, dtype=np.float32), MIMI_SAMPLE_RATE
        y48 = np.concatenate(cleaned)
        return _resample(y48, RNNOISE_SR, MIMI_SAMPLE_RATE), MIMI_SAMPLE_RATE
