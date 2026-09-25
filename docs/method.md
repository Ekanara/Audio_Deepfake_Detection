# Log-Likelihood_fake_only: Training, Inference, and Analysis

> **Log-Likelihood (LL)**: the log probability of a sample's embedding under a fitted Gaussian distribution.

## Model Architecture

BEATs backbone (pretrained audio encoder) with three output heads:
- **bonafide_head** (527-dim): Main embedding used for detection scoring
- **softmax_head** (N-class, already softmax-activated): Multi-class generator classification
- **contrastive_head** (256-dim): Used during training for contrastive learning

## Training: 2-Stage Pipeline

### Loss Function

```
L = 0.2 * ArcFace + 0.8 * CCL + ce_w * CE
```

- **ArcFace** (Angular Margin Loss): 5-class classification loss on bonafide_head. Pushes embeddings of different generator types apart with angular margins (m=3, s=30).
- **CCL** (Center Contrastive Loss): Binary real/fake contrastive loss on bonafide_head. Learns centers for real and fake classes, pulls same-class embeddings together and pushes different-class embeddings apart. Has a `real_w` parameter that controls how strongly real samples are pulled toward the real center.
- **CE** (Cross-Entropy): Multi-class classification loss on softmax_head. Uses weighted NLL loss (softmax_head is already softmax-activated, so we use `log(softmax) + nll_loss`, NOT `cross_entropy`). The real class gets higher weight (`ce_real_w`) to compensate for class imbalance.

### Stage 1: Feature Learning
- **Epochs**: 15
- **Learning rate**: 3e-7 (backbone), 3e-6 (loss heads, 10x multiplier)
- **CE weight**: 3.0
- **CE real weight**: 2.5
- **CCL real_w**: 4.0
- **ArcFace weight**: 0.2

Purpose: Learn discriminative embeddings from scratch using BEATs pretrained features. The backbone fine-tunes slowly while the classification/contrastive heads learn faster.

### Stage 2: Refinement
- **Epochs**: 8
- **Learning rate**: 1.5e-7 (backbone), 1.5e-6 (heads)
- **CE weight**: 3.0
- **CE real weight**: 4.0 (increased from 2.5)
- **CCL real_w**: 4.0
- **ArcFace weight**: 0.2

Purpose: Fine-tune from the best Stage 1 checkpoint with lower learning rate and stronger real-class emphasis. The increased `ce_real_w` (2.5 → 4.0) pushes the model to better separate real samples.

### Training Data
- **Balanced sampling**: Each batch contains 6 samples per fake generator + 24 real samples (total 48/batch for 5-class)
- **Data**: Event_train with 5 classes: real, fake_ata_01 (ATA-Audioldm1), fake_tta_01 (TTA-Audiogen), fake_tta_02 (TTA-Audioldm1), fake_tta_03 (TTA-Audioldm2)

## Inference: Log-Likelihood_fake_only Scoring

The inference method does NOT use any classification head. It uses only the 527-dim bonafide_head embeddings with a statistical scoring approach:

### Step 1: Build Fake Reference (training data only)
Collect bonafide_head embeddings from all **fake samples in the training set** (Event_train fake only) and fit a multivariate Gaussian distribution:
```
N(μ_fake, Σ_fake)
```
where μ_fake is the mean and Σ_fake is the covariance matrix of the fake training embeddings.

**Important**: No test set data is used to build this reference. The Gaussian is computed entirely from the training set's fake samples. This means the scoring is purely based on what the model learned about "fake" during training.

### Step 2: Score Test Samples
For each test sample, extract its bonafide_head embedding, then compute the negative log-likelihood under the fake Gaussian:
```
score(x) = -log p(x | N(μ_fake, Σ_fake))
```

- **High score** → embedding is far from the fake training distribution → classified as **real**
- **Low score** → embedding is close to the fake training distribution → classified as **fake**

The test samples are only scored — they do not influence the reference distribution in any way.

### Step 3: Threshold
Use the Equal Error Rate (EER) threshold from the ROC curve. At EER, the false positive rate equals the false negative rate, giving a balanced decision boundary.

### Why This Works
The model learns to map fake audio into a compact region of the 527-dim embedding space (driven by ArcFace + CCL during training). Real audio, being fundamentally different, lands in a different region. By measuring distance from the fake cluster, we effectively detect whether a sample "looks fake."

## Why Real Reference Scoring Fails

### test_track2 (9,974 samples: 1,994 real, 7,980 fake)

| Method | AUC | Acc | F1_macro |
|--------|-----|-----|----------|
| -Log-Likelihood(fake) [baseline] | 0.9288 | 85.52% | 0.8035 |
| Log-Likelihood(real) - Log-Likelihood(fake) | 0.9256 | 85.06% | 0.7979 |
| Log-Likelihood(real) only | 0.2559 | 32.03% | 0.2942 |
| Cosine similarity to real centroid | 0.9023 | 83.88% | 0.7840 |

Note: Mahalanobis distance to fake/real reference produces identical classification results to -Log-Likelihood(fake)/Log-Likelihood(real) because `-log p(x|N(μ,Σ)) = 0.5 * d_Mahalanobis(x)² + const` — the constant is fixed per reference, so sample rankings and EER thresholds are the same.

Confusion matrix for -Log-Likelihood(fake) baseline on test_track2:

|  | Predicted Fake | Predicted Real |
|--|---------------|---------------|
| **Actual Fake** (7,980) | TN = 6,822 (85.51%) | FP = 1,158 (14.49%) |
| **Actual Real** (1,994) | FN = 288 (14.44%) | TP = 1,706 (85.56%) |

### Event_test (14,408 samples: 1,801 real, 12,607 fake)

| Method | AUC | Acc | F1_macro |
|--------|-----|-----|----------|
| -Log-Likelihood(fake) [baseline] | 0.9443 | 88.13% | 0.7893 |
| Log-Likelihood(real) - Log-Likelihood(fake) | 0.9569 | 88.58% | 0.7956 |
| Log-Likelihood(real) only | 0.6748 | 61.61% | 0.5120 |
| Cosine similarity to real centroid | 0.9459 | 86.35% | 0.7649 |

On Event_test, Log-Likelihood(real) only reaches 61.61% — better than test_track2's 32% because Event_test real audio partially overlaps with Event_train real (both from TUT/UrbanSound8K domain), but still far below the baseline.

### TUTASC19_test (14,408 samples: 1,801 real, 12,607 fake)

| Method | AUC | Acc | F1_macro |
|--------|-----|-----|----------|
| -Log-Likelihood(fake) [baseline] | 0.9907 | 95.43% | 0.9079 |
| Log-Likelihood(real) - Log-Likelihood(fake) | 0.9918 | 95.90% | 0.9150 |
| Log-Likelihood(real) only | 0.8790 | 81.62% | 0.7061 |
| Cosine similarity to real centroid | 0.9834 | 93.62% | 0.8741 |

On TUTASC19_test, all methods perform best. The likelihood ratio reaches 95.90%, and even Log-Likelihood(real) only achieves 81.62% — confirming that TUTASC19 real audio is closest to Event_train real distribution.

