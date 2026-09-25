"""Per-dataset metrics for Combined model on ENVSDD — single global threshold."""
import os, sys, json, numpy as np, torch, pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
from argparse import Namespace
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from base_dataset import BeatsDataset
from beats.model_beat import model_beat

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

def load_model(p):
    ckpt = torch.load(p, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    pipeline = model_beat(num_label=5, three_loss=True, feature_layer="predictor").to(device)
    ps = {k[9:]: v for k, v in sd.items() if k.startswith("pipeline.")}
    cur = pipeline.state_dict()
    pipeline.load_state_dict({k: v for k, v in ps.items() if k in cur and v.shape == cur[k].shape}, strict=False)
    return pipeline

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

# Load metadata for wavename → dataset mapping
meta = pd.read_csv("data/test/test_set/test_metadata.csv")
meta_map = {}
for _, row in meta.iterrows():
    wb = row["wavename"].replace(".wav", "")
    meta_map[wb] = {
        "dataset": row["source dataset"],
        "gen_combo": row["faketype"] + "/" + row["generator"],
    }

# Load model
print("Loading Combined model...")
pipeline = load_model("checkpoint/Beats_journal/combined_Event_TUTASC19_scratch/stage2/S2r_ep02.ckpt")

# Collect reference
print("Collecting reference features...")
ref_bn, _, _ = collect(pipeline, "data/label/beats/Event_train_stage1_fakeonly.json")
gF = fit_gaussian(ref_bn)
del ref_bn
print("Reference ready.")

# Step 1: Score all ENVSDD to get global threshold
print("Collecting ENVSDD test features (39768 samples)...")
test_bn, test_lbl, inv = collect(pipeline, "data/label/beats/ENVSDD_test.json")
real_idx = {v: k for k, v in inv.items()}["real"]
binary = (test_lbl == real_idx).astype(int)
scores = -loglik(test_bn, gF)

fpr, tpr, thrs = roc_curve(binary, scores, pos_label=1)
fnr = 1.0 - tpr
eer_idx = np.argmin(np.abs(fpr - fnr))
eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
global_thr = thrs[eer_idx]
preds = (scores >= global_thr).astype(int)

overall_auc = roc_auc_score(binary, scores)
overall_acc = accuracy_score(binary, preds)
overall_f1r = f1_score(binary, preds, pos_label=1)
overall_f1f = f1_score(binary, preds, pos_label=0)
total_fake_wrong = ((binary == 0) & (preds == 1)).sum()
total_real_wrong = ((binary == 1) & (preds == 0)).sum()

print(f"\n{'='*100}")
print(f"  OVERALL ENVSDD: {len(binary)} samples | AUC={overall_auc:.4f} | EER={eer*100:.2f}% | Acc={overall_acc*100:.2f}%")
print(f"  F1(real)={overall_f1r:.4f} | F1(fake)={overall_f1f:.4f} | F1(macro)={(overall_f1r+overall_f1f)/2:.4f}")
print(f"  Fake wrong: {total_fake_wrong} | Real wrong: {total_real_wrong} | Global threshold: {global_thr:.2f}")
print(f"{'='*100}")

del test_bn, test_lbl, scores, binary, preds

# Step 2: Score each per-dataset JSON with the GLOBAL threshold
out_dir = "data/label/beats/per_dataset"
os.makedirs(out_dir, exist_ok=True)

# Create per-dataset JSONs
for ds_name in sorted(meta["source dataset"].unique()):
    sub = meta[meta["source dataset"] == ds_name]
    entries = []
    for _, row in sub.iterrows():
        label = "real" if row["faketype"] == "real" else "fake"
        entries.append({"audio": f"data/test/test_set/audio/{row['wavename']}", "label": label})
    path = os.path.join(out_dir, f"{ds_name.replace(' ', '_')}.json")
    with open(path, "w") as f:
        json.dump(entries, f)

print(f"\n{'Dataset':20s} {'N':>6s} {'AUC':>7s} {'Acc%':>7s} {'F1r':>7s} {'F1f':>7s} {'F1m':>7s} {'FakeWrong':>10s} {'RealWrong':>10s}")
print("-" * 100)

all_gen_scores = []  # collect for generator breakdown

for ds_name in sorted(meta["source dataset"].unique()):
    path = os.path.join(out_dir, f"{ds_name.replace(' ', '_')}.json")
    bn, lbl, inv2 = collect(pipeline, path)
    real_idx2 = {v: k for k, v in inv2.items()}["real"]
    b = (lbl == real_idx2).astype(int)
    s = -loglik(bn, gF)
    p = (s >= global_thr).astype(int)

    # Also store for generator breakdown
    sub = meta[meta["source dataset"] == ds_name]
    for i, (_, row) in enumerate(sub.iterrows()):
        if row["faketype"] != "real":
            gc = row["faketype"] + "/" + row["generator"]
            all_gen_scores.append({"gen_combo": gc, "is_fake_wrong": int(b[i] == 0 and p[i] == 1)})

    if len(np.unique(b)) < 2:
        n = len(b)
        fw = ((b == 0) & (p == 1)).sum()
        rw = ((b == 1) & (p == 0)).sum()
        acc = accuracy_score(b, p)
        print(f"{ds_name:20s} {n:6d}    n/a {acc*100:6.2f}%     n/a     n/a     n/a {fw:10d} {rw:10d}")
        continue

    auc = roc_auc_score(b, s)
    acc = accuracy_score(b, p)
    f1r = f1_score(b, p, pos_label=1)
    f1f = f1_score(b, p, pos_label=0)
    f1m = (f1r + f1f) / 2
    fw = ((b == 0) & (p == 1)).sum()
    rw = ((b == 1) & (p == 0)).sum()
    print(f"{ds_name:20s} {len(b):6d} {auc:7.4f} {acc*100:6.2f}% {f1r:7.4f} {f1f:7.4f} {f1m:7.4f} {fw:10d} {rw:10d}")

# Generator breakdown
gen_df = pd.DataFrame(all_gen_scores)
print(f"\n{'Generator':20s} {'N_fake':>7s} {'FakeWrong':>10s} {'ErrRate':>8s}")
print("-" * 50)
for gc in sorted(gen_df["gen_combo"].unique()):
    sub = gen_df[gen_df["gen_combo"] == gc]
    n = len(sub)
    fw = sub["is_fake_wrong"].sum()
    print(f"{gc:20s} {n:7d} {fw:10d} {fw/n*100:7.2f}%")

print("\nDone!")
