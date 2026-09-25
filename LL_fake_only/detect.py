"""
detect.py — real/fake detection for a single .wav (no distribution recompute)
=============================================================================

Loads the trained BEATs model + the PRE-COMPUTED fake Gaussian and calibration
from reference_stats.npz, then scores any .wav/.flac file. Does NOT re-embed
the 29k fake training samples — only the input file is embedded, so it runs
in seconds (after the one-time model load).

Quick start
-----------
  # Score the bundled samples
  python LL_fake_only/detect.py LL_fake_only/samples/sample_real.wav
  python LL_fake_only/detect.py LL_fake_only/samples/sample_fake.wav

  # Score your own files
  python LL_fake_only/detect.py my_audio.wav
  python LL_fake_only/detect.py a.wav b.wav c.wav

  # Long audio: average over all 4s segments
  python LL_fake_only/detect.py my_audio.wav --all_segments

Output: prediction (REAL/FAKE) + confidence score for each file.

If reference_stats.npz is missing, run once:
  python LL_fake_only/precompute_reference.py
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.linalg import cho_factor, cho_solve

_HERE = os.path.dirname(os.path.abspath(__file__))
# Prefer the bundled src/ (standalone package); fall back to the repo's ../src.
for _cand in (os.path.join(_HERE, "src"), os.path.join(_HERE, "..", "src")):
    _cand = os.path.abspath(_cand)
    if os.path.isdir(_cand):
        sys.path.insert(0, os.path.dirname(_cand))  # parent → enables `import src.xxx`
        sys.path.insert(0, _cand)                    # src/   → enables `from base_dataset import`
        break
from inference.inference_ccl import load_pipeline_and_ccl

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HERE = _HERE
DEFAULT_CKPT = os.path.join(HERE, "checkpoint", "ll_fake_only_stage2.ckpt")
DEFAULT_STATS = os.path.join(HERE, "reference_stats.npz")

# Audio params (must match training)
SAMPLE_RATE = 16000
SEGMENT_SEC = 4
SEGMENT_LEN = SAMPLE_RATE * SEGMENT_SEC  # 64000


# ─────────────────────────────────────────────────────────────────────────────
# Audio preprocessing (mirrors BeatsDataset.get_file_content)
# ─────────────────────────────────────────────────────────────────────────────
def load_audio_segments(file_path):
    wav, orig_sr = sf.read(file_path)
    if wav.ndim > 1:
        wav = wav[:, 0]
    if orig_sr != SAMPLE_RATE:
        wav = librosa.core.resample(wav, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
    wav = np.asarray(wav, dtype=np.float32)
    n = len(wav)
    if n < SEGMENT_LEN:
        while len(wav) < SEGMENT_LEN:
            wav = np.concatenate([wav, wav])
        n = len(wav)
    split_num = int(2 + np.floor((n - SEGMENT_LEN) * 2 / SEGMENT_LEN))
    segs = []
    for m in range(split_num):
        if m == split_num - 1:
            t0, t1 = n - SEGMENT_LEN, n
        else:
            t0 = int(m * SEGMENT_LEN / 2)
            t1 = t0 + SEGMENT_LEN
        segs.append(wav[t0:t1])
    return np.stack(segs)


# ─────────────────────────────────────────────────────────────────────────────
# Detector — loads model + precomputed stats once, scores many files
# ─────────────────────────────────────────────────────────────────────────────
class Detector:
    def __init__(self, checkpoint=DEFAULT_CKPT, stats_path=DEFAULT_STATS, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.isfile(stats_path):
            raise FileNotFoundError(
                f"{stats_path} not found.\n"
                f"Run once to generate it:\n"
                f"  python LL_fake_only/precompute_reference.py")

        # Load precomputed Gaussian + calibration
        s = np.load(stats_path)
        self.mu = s["mu"]
        self.logdet = float(s["logdet"])
        self.threshold = float(s["threshold"])
        self.scale = float(s["scale"])
        self.cf, self.lower = cho_factor(s["cov"])
        logger.info(f"Loaded stats: threshold={self.threshold:.2f} scale={self.scale:.2f}")

        # Load model (needed to embed input)
        print(f"Loading model... ", end="", flush=True)
        self.pipeline, _, _ = load_pipeline_and_ccl(
            checkpoint_path=checkpoint, device=self.device, mode="beats",
            num_classes=5, embed_dim=527, truncate_layers=0, beats_feature="predictor")
        self.pipeline.eval()
        print("done.")

    @torch.no_grad()
    def _embed(self, segments):
        audio = torch.from_numpy(segments).float().to(self.device)
        out = self.pipeline.forward_pipeline(audio)
        emb = out[0] if isinstance(out, tuple) else out
        return emb.float().cpu().numpy()

    def _score(self, embs):
        """-log p(x|fake) up to const: high = far from fake = real."""
        diff = embs.astype(np.float64) - self.mu
        solved = cho_solve((self.cf, self.lower), diff.T).T
        mahal_sq = np.einsum("ij,ij->i", diff, solved)
        return 0.5 * (mahal_sq + self.logdet)

    def detect(self, wav_path, all_segments=False):
        segs = load_audio_segments(wav_path)
        if not all_segments:
            segs = segs[:1]
        embs = self._embed(segs)
        score = float(self._score(embs).mean())

        # Calibrated probability of being real
        p_real = 1.0 / (1.0 + np.exp(-(score - self.threshold) / self.scale))
        prediction = "real" if score >= self.threshold else "fake"
        confidence = p_real if prediction == "real" else 1.0 - p_real

        return {
            "file": str(wav_path),
            "prediction": prediction,
            "confidence": float(confidence),
            "p_real": float(p_real),
            "score": score,
            "threshold": self.threshold,
            "n_segments": len(segs),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Detect real/fake for a single .wav using precomputed stats",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help=".wav or .flac file(s) to score")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--stats", default=DEFAULT_STATS)
    p.add_argument("--all_segments", action="store_true",
                   help="Average over all 4s segments (default: first only)")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    det = Detector(checkpoint=args.checkpoint, stats_path=args.stats, device=args.device)

    print()
    print(f"{'File':<40} {'Prediction':>10} {'Confidence':>11} {'Score':>10}")
    print("-" * 75)
    for wav in args.files:
        if not os.path.isfile(wav):
            logger.warning(f"Not found: {wav}")
            continue
        r = det.detect(wav, all_segments=args.all_segments)
        flag = "✓" if r["prediction"] == "real" else "✗"
        print(f"{Path(wav).name:<40} {r['prediction'].upper():>9}{flag} "
              f"{r['confidence']*100:>9.1f}% {r['score']:>10.1f}")
    print()


if __name__ == "__main__":
    main()
