# LL_fake_only: Log-Likelihood Fake-Only Scoring for Audio Deepfake Detection

Two-stage BEATs training + statistical inference for the ESDD challenge (ICASSP).
Scores audio as **real** or **fake** by measuring distance from a Gaussian fitted on fake training embeddings — no classification head used at inference time.

> **TL;DR — just want to test one .wav?**
> ```bash
> python LL_fake_only/detect.py LL_fake_only/samples/sample_fake.wav
> ```
> The fake distribution is pre-computed and bundled (`reference_stats.npz`), so this only embeds your one file — no need to re-run the test set.

---

## Files

This folder is **fully self-contained** — it bundles all model code (`src/`), so it can be copied anywhere and run on its own.
The two large weight files are the only exception: they are **not tracked in git** (GitHub caps files at 100 MB), so a fresh clone needs them added manually — see [Prerequisites](#prerequisites--do-this-before-running) step 4.

```
LL_fake_only/
├── detect.py             ★ Score a single .wav → REAL/FAKE + confidence (FAST)
├── precompute_reference.py  One-time: builds reference_stats.npz (already bundled)
├── reference_stats.npz   Pre-computed fake Gaussian + EER threshold + calibration
├── samples/
│   ├── sample_real.wav   A real sample (from test_track2)
│   └── sample_fake.wav   A fake sample (from test_track2)
├── infer.sh / infer.py   Full evaluation on test sets (metrics + plots + CSV)
├── train.sh / train.py   2-stage training
├── requirements.txt
├── data/                 Training + test JSON manifests
│   ├── Event_train_stage1.json           Full training set (5-class)
│   ├── Event_train_stage1_fakeonly.json  Fake-only subset — Gaussian reference
│   ├── Event_train_stage1_realonly.json  Real-only subset
│   ├── test_track2.json                  Challenge test set (9,974 samples)
│   ├── Event_test_5class.json            Event test set (14,408 samples)
│   └── TUTASC19_test_5class.json         TUTASC19 test set (14,408 samples)
├── checkpoint/
│   └── ll_fake_only_stage2.ckpt          Best Stage 2 model (1.1 GB, not in git)
└── src/                  Bundled model code (no external repo needed)
    ├── base_dataset.py  base_pipeline.py  base_ptln.py
    ├── beats/           BEATs backbone + BEATs_..._cpt2.pt (347 MB, not in git)
    ├── inference/       inference_ccl.py + _lib/
    └── utils/           arguments.py  training_utils.py
```

Commands can be run from anywhere — every path resolves relative to this folder. Examples below use the `LL_fake_only/...` prefix (running from the repo root); if you `cd LL_fake_only` first, drop the prefix.

---

## Prerequisites — do this before running

**1. Python 3.11** (tested on 3.11.0)
```bash
python --version          # should print Python 3.11.x
```

**2. Create an environment and install dependencies**
```bash
conda create -n ll_fake_only python=3.11 -y
conda activate ll_fake_only
pip install -r LL_fake_only/requirements.txt
```
> **GPU torch:** plain `pip install torch` may give a CPU-only build. For GPU, install a CUDA build matching your driver (tested on `torch 2.9.1+cu126`) — see https://pytorch.org for the exact command.

**3. GPU is recommended but not required**
`detect.py` auto-detects CUDA and falls back to CPU. CPU works but the model load (~20s) and embedding are slower. Force a device with `--device cuda` or `--device cpu`.

**4. Put the two large weight files in place**
Nothing runs without these. They exceed GitHub's 100 MB file limit, so they are **excluded from the repository** — a packaged copy of this folder (zip / shared drive) already contains them, a `git clone` does not.

| File | Size | Where to get it |
|------|------|-----------------|
| `checkpoint/ll_fake_only_stage2.ckpt` | 1.1 GB | trained Stage 2 checkpoint (shared separately, or produced by `train.sh`) |
| `src/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt` | 347 MB | [official BEATs release](https://github.com/microsoft/unilm/tree/master/beats) — `BEATs_iter3+ (AS2M) (cpt2)` |

Check they are present and full size:
```bash
ls -lh LL_fake_only/checkpoint/*.ckpt LL_fake_only/src/beats/*.pt
```
If a file is missing or only a few KB, `detect.py` will fail to load. Note that a checkpoint you train yourself needs `precompute_reference.py` re-run (step 2) so the bundled distribution matches it.

**5. Input audio format**
WAV or FLAC, mono or stereo, **any** sample rate (auto-resampled to 16 kHz) and **any** length (auto-segmented to 4-second windows). No manual preprocessing needed.

**6. Verify the setup**
```bash
python LL_fake_only/detect.py LL_fake_only/samples/sample_fake.wav
```
Expected: `FAKE✗  ~78%`. If you see that, you're ready.

---

## How to Use

### ★ 1. Detect real/fake on a single .wav  _(what you usually want)_

`detect.py` loads the model + the **pre-computed** fake distribution from `reference_stats.npz`, then embeds only your input file. It does **not** re-score the 29k fake training samples or the test set.

```bash
# Bundled samples (sanity check)
python LL_fake_only/detect.py LL_fake_only/samples/sample_real.wav
python LL_fake_only/detect.py LL_fake_only/samples/sample_fake.wav

# Your own file(s)
python LL_fake_only/detect.py my_audio.wav
python LL_fake_only/detect.py a.wav b.wav c.wav

# Long audio — average score over all 4s segments
python LL_fake_only/detect.py my_audio.wav --all_segments
```

**Output:**
```
File                                     Prediction  Confidence      Score
---------------------------------------------------------------------------
sample_real.wav                               REAL✓      99.2%     6216.1
sample_fake.wav                               FAKE✗      78.0%     1707.6
```

- **Prediction** — `REAL` if score ≥ threshold, else `FAKE`
- **Confidence** — calibrated probability of the predicted class (50–100%)
- **Score** — `-log p(x | N(μ_fake, Σ_fake))`; higher = farther from fake = more real

Everything (Gaussian, EER threshold, confidence calibration) is baked into `reference_stats.npz`. You only pay the one-time model load (~20s), then each file scores in well under a second.

**Options:** `--all_segments` (average over 4s windows), `--checkpoint PATH`, `--stats PATH`, `--device cuda/cpu`.

---

### 2. Re-build the reference (only if you retrain)

`reference_stats.npz` is already bundled, so **you normally skip this**. Re-run it only if you train a new checkpoint and want the distribution to match:

```bash
python LL_fake_only/precompute_reference.py
```

This embeds the fake reference (`Event_train_stage1_fakeonly.json`), fits the Gaussian, computes the EER threshold + confidence calibration on `test_track2.json`, and saves `reference_stats.npz`.

---

### 3. Full evaluation on test sets

For metrics (AUC / Acc / F1), confusion matrices, ROC curves, and per-audio CSV across whole test sets:

```bash
# Shell — all 3 bundled test sets
bash LL_fake_only/infer.sh

# Python — one set with plots + CSV
python LL_fake_only/infer.py --plot --save_csv

# Multiple sets
python LL_fake_only/infer.py \
    --test_json LL_fake_only/data/test_track2.json \
                LL_fake_only/data/Event_test_5class.json \
                LL_fake_only/data/TUTASC19_test_5class.json \
    --plot --save_csv
```

Outputs → `inference_outputs/ll_fake_only/`:
```
confusion_roc_<set>.png    Confusion matrix + ROC curve
score_dist_<set>.png       Score distribution (real vs fake)
per_audio_<set>.csv        Per-file scores and predictions
```

Console example (test_track2):
```
=======================================================
LL_fake_only — test_track2
=======================================================
  AUC       : 0.9288
  Accuracy  : 85.52%
  F1 macro  : 0.8035
  EER       : 14.49%  (thr=2650.56)
=======================================================
```

---

### 4. Train from scratch

> Skip this if you use the bundled checkpoint.

```bash
# Both stages (shell)
bash LL_fake_only/train.sh both
WANDB_API=your_key bash LL_fake_only/train.sh both

# Or Python with custom hyperparameters
python LL_fake_only/train.py --stage both \
    --learning_rate 5e-7 --ce_w 2.0 --ce_real_w 3.0
```

Checkpoints → `checkpoint/ll_fake_only_stage1/` and `checkpoint/ll_fake_only_stage2/`.
After training, re-run `precompute_reference.py` (step 2) so `detect.py` uses the matching distribution.

**Hyperparameters:**

| Argument | Stage 1 | Stage 2 | Description |
|----------|---------|---------|-------------|
| `--learning_rate` | `3e-7` | `1.5e-7` | Backbone LR |
| `--num_train_epochs` | `15` | `8` | Epochs |
| `--ce_w` | `3.0` | `3.0` | Cross-entropy weight |
| `--ce_real_w` | `2.5` | `4.0` | Real-class weight in CE |
| `--ccl_real_w` | `4.0` | `4.0` | Real-class weight in CCL |

---

## Method

### Scoring

```
score(x) = -log p(x | N(μ_fake, Σ_fake))
         = 0.5 · d_Mahalanobis(x, μ_fake)² + 0.5 · log|Σ_fake| + const
```

- **High score** → far from fake cluster → **Real**
- **Low score** → inside fake cluster → **Fake**
- Decision at EER threshold; confidence = sigmoid((score − threshold) / scale)

### Training losses

```
L = 0.2 × ArcFace + 0.8 × CCL + ce_w × CE
```

| Loss | Role |
|------|------|
| **ArcFace** | 5-class angular margin — separates each fake generator |
| **CCL** | Binary real/fake center contrastive loss |
| **CE** | Multi-class on softmax head, upweighted toward real |

### Why -LL(fake) beats LL(real)

The model packs all fake audio into a tight cluster; real audio lands far away. Distance from the fake cluster detects "fakeness" independent of the real audio's domain. LL(real) fails cross-domain: its real Gaussian is fitted on training real (TUT/UrbanSound8K), so test real (e.g. VGGSound in test_track2) falls outside it and gets misread as fake.

---

## Results

| Test set | AUC | Acc | F1_macro |
|----------|-----|-----|----------|
| test_track2 (1,994 real / 7,980 fake) | 0.9288 | 85.52% | 0.8035 |
| Event_test (1,801 real / 12,607 fake) | 0.9443 | 88.13% | 0.7893 |
| TUTASC19_test (1,801 real / 12,607 fake) | 0.9907 | 95.43% | 0.9079 |

All using `-LL(fake)` scoring with the bundled Stage 2 checkpoint.

---

## Data Format

```json
[
  {"audio": "/path/to/audio.wav", "label": "real"},
  {"audio": "/path/to/audio.wav", "label": "fake_tta_01"}
]
```

**5-class labels:** `real`, `fake_ata_01` (ATA-Audioldm1), `fake_tta_01` (TTA-Audiogen), `fake_tta_02` (TTA-Audioldm1), `fake_tta_03` (TTA-Audioldm2).
**Audio:** WAV/FLAC, any sample rate (→16 kHz), any length (→4s windows, 50% overlap).

---

## Implementation Notes

- `softmax_head` output is already softmax'd — use `F.nll_loss(log(probs))`, not `F.cross_entropy`
- `num_workers=0` required on WSL; use `tmux` for long training jobs
- All scoring uses the 527-dim `bonafide_head` (predictor output)
- `reference_stats.npz` stores: `mu`, `cov`, `logdet`, `threshold`, `scale`, `auc`, `eer` — regenerate it whenever the checkpoint changes
