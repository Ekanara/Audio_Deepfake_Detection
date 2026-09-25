"""Generate confusion matrix + ROC for -LL(fake) baseline on test_track2.
Runs the full inference flow and uses the project's own plotting function."""
import os, sys, numpy as np, torch
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from argparse import Namespace
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from base_dataset import BeatsDataset
from inference.inference_ccl import load_pipeline_and_ccl

device = "cuda"

def fit_gaussian(X, eps=1e-6):
    X = X.astype(np.float64)
    mu = X.mean(0)
    cov = np.cov(X, rowvar=False) + eps * np.eye(X.shape[1])
    cf, lower = cho_factor(cov)
    _, logdet = np.linalg.slogdet(cov)
    return mu, cf, lower, logdet

def loglik(X, g):
    mu, cf, lower, logdet = g
    diff = X.astype(np.float64) - mu
    solved = cho_solve((cf, lower), diff.T).T
    return -0.5 * (np.einsum("ij,ij->i", diff, solved) + logdet)

@torch.no_grad()
def collect(pipeline, json_file):
    ds_args = Namespace(num_label=5, three_loss=True, audio_aug=False,
                        audio_mixup=False, audio_aug_prob=0, audio_mixup_prob=0)
    ds = BeatsDataset(json_file=json_file, transformation=None, args=ds_args)
    inv = {v: k for k, v in ds.label.items()}
    dl = DataLoader(ds, batch_size=512, num_workers=0, shuffle=False, pin_memory=True)
    all_bn, all_lbl = [], []
    pipeline.eval()
    for batch in dl:
        a = batch["audio"].to(device, dtype=torch.float32)
        o = pipeline.forward_pipeline(a)
        all_bn.append(o[0].float().cpu().numpy())
        all_lbl.extend(batch["label"].cpu().numpy())
        del a, o
    return np.concatenate(all_bn), np.array(all_lbl), inv

# ---------- run ----------
ckpt = "checkpoint/Beats_journal/Beats_Event_2stage_scratch_stage2_LR1.5e-7_ce3_realW4_8epoch/sample-03.ckpt"
print("Loading checkpoint...")
pipeline, _, _ = load_pipeline_and_ccl(
    checkpoint_path=ckpt, device=device, mode="beats",
    num_classes=5, embed_dim=527, truncate_layers=0, beats_feature="predictor")

print("Collecting fake reference...")
ref_fake_bn, _, _ = collect(pipeline, "data/label/beats/Event_train_stage1_fakeonly.json")
gF = fit_gaussian(ref_fake_bn)

print("Collecting test_track2...")
test_bn, test_lbl, inv = collect(pipeline, "data/label/beats/test_track2.json")
real_idx = {v: k for k, v in inv.items()}["real"]
binary = (test_lbl == real_idx).astype(int)  # 1=real, 0=fake

scores = -loglik(test_bn, gF)  # -LL(fake): high = real

# EER threshold
fpr, tpr, thrs = roc_curve(binary, scores, pos_label=1)
fnr = 1.0 - tpr
eer_idx = np.argmin(np.abs(fpr - fnr))
eer = fpr[eer_idx]
eer_thr = thrs[eer_idx]
auc = roc_auc_score(binary, scores)

predictions = (scores >= eer_thr).astype(int)
cm = confusion_matrix(binary, predictions, labels=[0, 1])

print(f"AUC={auc:.4f}  EER={eer:.4f}  thr={eer_thr:.4f}")
print(f"Confusion matrix:\n{cm}")

# ---------- plot (same format as project's plot_confusion_and_roc) ----------
class_names = ["fake", "real"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
annot_labels = np.array([[f"{v:.2f}" for v in row] for row in cm_pct])
sns.heatmap(cm_pct, annot=annot_labels, fmt='s', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 11}, ax=axes[0])
axes[0].set_title('Confusion Matrix', fontsize=14)
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

axes[1].plot(fpr, tpr, color='steelblue',
             label=f'ROC Curve (AUC = {auc:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[1].scatter([fpr[eer_idx]], [tpr[eer_idx]], color='red', zorder=5, s=80,
                label=f'EER = {eer:.3f}')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate', fontsize=12)
axes[1].set_title('ROC Curves', fontsize=14)
axes[1].legend(loc='lower right', fontsize=10)
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
out = "inference_outputs/confusion_matrix_test_track2.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
