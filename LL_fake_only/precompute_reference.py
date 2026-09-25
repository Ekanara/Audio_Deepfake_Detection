"""
precompute_reference.py — bake the fake Gaussian + calibration into one file
============================================================================

Run this ONCE to generate `reference_stats.npz`, which stores everything
detect.py needs to score a single .wav without re-embedding the 29k fake
training samples every time:

  - mu, cov, logdet  : fake Gaussian N(mu_fake, Sigma_fake)
  - threshold        : EER decision threshold (from test_track2)
  - scale            : logistic calibration scale for the confidence score

After this runs, detect.py loads reference_stats.npz instantly and only
needs to embed the one input file.

Usage
-----
  python LL_fake_only/precompute_reference.py
"""
import logging
import os
import sys

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
# Prefer the bundled src/ (standalone package); fall back to the repo's ../src.
for _cand in (os.path.join(_HERE, "src"), os.path.join(_HERE, "..", "src")):
    _cand = os.path.abspath(_cand)
    if os.path.isdir(_cand):
        sys.path.insert(0, os.path.dirname(_cand))  # parent → enables `import src.xxx`
        sys.path.insert(0, _cand)                    # src/   → enables `from base_dataset import`
        break
from argparse import Namespace
from base_dataset import BeatsDataset
from inference.inference_ccl import load_pipeline_and_ccl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT = os.path.join(_HERE, "checkpoint", "ll_fake_only_stage2.ckpt")
FAKE_REF   = os.path.join(_HERE, "data", "Event_train_stage1_fakeonly.json")
CALIB_JSON = os.path.join(_HERE, "data", "test_track2.json")  # labeled set for threshold + calibration
OUT_PATH   = os.path.join(_HERE, "reference_stats.npz")


@torch.no_grad()
def embed_json(pipeline, json_file, device, batch_size=512):
    ds_args = Namespace(num_label=5, three_loss=True, audio_aug=False,
                        audio_mixup=False, audio_aug_prob=0, audio_mixup_prob=0)
    ds = BeatsDataset(json_file=json_file, transformation=None, args=ds_args)
    label_map = ds.label
    dl = DataLoader(ds, batch_size=batch_size, num_workers=0,
                    shuffle=False, pin_memory=True)
    embs, labels = [], []
    for batch in dl:
        audio = batch["audio"].to(device, dtype=torch.float32)
        out = pipeline.forward_pipeline(audio)
        emb = out[0] if isinstance(out, tuple) else out
        embs.append(emb.float().cpu().numpy())
        labels.extend(batch["label"].cpu().numpy())
    return np.concatenate(embs), np.array(labels), label_map


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load model
    logger.info(f"Loading checkpoint: {CHECKPOINT}")
    pipeline, _, _ = load_pipeline_and_ccl(
        checkpoint_path=CHECKPOINT, device=device, mode="beats",
        num_classes=5, embed_dim=527, truncate_layers=0, beats_feature="predictor")

    # 2. Fit fake Gaussian
    logger.info(f"Embedding fake reference: {FAKE_REF}")
    fake_emb, _, _ = embed_json(pipeline, FAKE_REF, device)
    X = fake_emb.astype(np.float64)
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1])
    _, logdet = np.linalg.slogdet(cov)
    logger.info(f"Gaussian fitted: {len(X)} samples, dim={X.shape[1]}, logdet={logdet:.2f}")

    # 3. Score calibration set → EER threshold + confidence scale
    logger.info(f"Embedding calibration set: {CALIB_JSON}")
    cal_emb, cal_lbl, label_map = embed_json(pipeline, CALIB_JSON, device)
    real_idx = label_map["real"]
    binary = (cal_lbl == real_idx).astype(int)  # 1=real, 0=fake

    cf, lower = cho_factor(cov)
    diff = cal_emb.astype(np.float64) - mu
    solved = cho_solve((cf, lower), diff.T).T
    mahal_sq = np.einsum("ij,ij->i", diff, solved)
    scores = 0.5 * (mahal_sq + logdet)          # = -log p(x|fake) up to const; high = real

    fpr, tpr, thr = roc_curve(binary, scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    threshold = float(thr[eer_idx])
    eer = float(fpr[eer_idx])
    auc = roc_auc_score(binary, scores)
    logger.info(f"AUC={auc:.4f}  EER={eer*100:.2f}%  threshold={threshold:.2f}")

    # Confidence calibration: P(real) = sigmoid((score - threshold)/scale)
    # 1-param logistic fit (center fixed at EER threshold, balanced classes).
    x = (scores - threshold).reshape(-1, 1)
    lr = LogisticRegression(fit_intercept=False, class_weight="balanced", C=1e6)
    lr.fit(x, binary)
    scale = float(1.0 / lr.coef_[0, 0])
    logger.info(f"Calibration scale={scale:.2f}")

    # 4. Save everything
    np.savez_compressed(
        OUT_PATH,
        mu=mu, cov=cov, logdet=logdet,
        threshold=threshold, scale=scale,
        auc=auc, eer=eer,
        n_fake_ref=len(X),
    )
    logger.info(f"Saved → {OUT_PATH}")

    # quick sanity check on the two stored stats
    p_at_thr = 1.0 / (1.0 + np.exp(-(0.0) / scale))
    logger.info(f"Sanity: P(real) at threshold = {p_at_thr:.3f} (should be 0.5)")


if __name__ == "__main__":
    main()
