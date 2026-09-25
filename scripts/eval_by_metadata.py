"""
Analyze LL_fake_only errors by generator and source dataset using test_metadata.csv.
"""
import os, sys, json, numpy as np, torch, pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import roc_auc_score, roc_curve
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

def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    pipeline = model_beat(num_label=5, three_loss=True, feature_layer="predictor").to(device)
    pipe_state = {k[9:]: v for k, v in state_dict.items() if k.startswith("pipeline.")}
    cur = pipeline.state_dict()
    compat = {k: v for k, v in pipe_state.items() if k in cur and v.shape == cur[k].shape}
    pipeline.load_state_dict(compat, strict=False)
    return pipeline

@torch.no_grad()
def collect_with_paths(pipeline, json_file):
    """Collect features + keep track of file paths."""
    ds_args = Namespace(num_label=5, three_loss=True, audio_aug=False,
                        audio_mixup=False, audio_aug_prob=0, audio_mixup_prob=0)
    ds = BeatsDataset(json_file=json_file, transformation=None, args=ds_args)
    inv = {v: k for k, v in ds.label.items()}
    dl = DataLoader(ds, batch_size=512, num_workers=4, shuffle=False, pin_memory=True)

    # Get paths from JSON directly (dataset doesn't return them)
    with open(json_file) as f:
        entries = json.load(f)
    paths = [e["audio"] for e in entries]

    all_bn, all_lbl = [], []
    pipeline.eval()
    for batch in dl:
        audio = batch["audio"].to(device, dtype=torch.float32)
        outputs = pipeline.forward_pipeline(audio)
        all_bn.append(outputs[0].float().cpu().numpy())
        all_lbl.extend(batch["label"].cpu().numpy())
        del audio, outputs
    return np.concatenate(all_bn), np.array(all_lbl), inv, paths

@torch.no_grad()
def collect(pipeline, json_file):
    ds_args = Namespace(num_label=5, three_loss=True, audio_aug=False,
                        audio_mixup=False, audio_aug_prob=0, audio_mixup_prob=0)
    ds = BeatsDataset(json_file=json_file, transformation=None, args=ds_args)
    dl = DataLoader(ds, batch_size=512, num_workers=4, shuffle=False, pin_memory=True)
    all_bn = []
    pipeline.eval()
    for batch in dl:
        audio = batch["audio"].to(device, dtype=torch.float32)
        outputs = pipeline.forward_pipeline(audio)
        all_bn.append(outputs[0].float().cpu().numpy())
        del audio, outputs
    return np.concatenate(all_bn)


def main():
    # Load metadata
    meta = pd.read_csv("data/test/test_set/test_metadata.csv")
    meta["wavename_base"] = meta["wavename"].str.replace(".wav", "", regex=False)

    ckpt_path = "checkpoint/Beats_journal/Beats_Event_2stage_scratch_stage2_LR1.5e-7_ce3_realW4_8epoch/sample-03.ckpt"
    test_json = "data/label/beats/Event_test_5class.json"

    print("Loading LL_fake_only (original)...")
    pipeline = load_model(ckpt_path)

    # Collect reference features
    ref_fake_bn = collect(pipeline, "data/label/beats/Event_train_stage1_fakeonly.json")
    ref_real_bn = collect(pipeline, "data/label/beats/Event_train_stage1_realonly.json")

    # Collect test features with paths
    test_bn, test_lbl, inv, paths = collect_with_paths(pipeline, test_json)
    real_idx = {v: k for k, v in inv.items()}["real"]

    # Fit Gaussian, compute scores
    gF = fit_gaussian(ref_fake_bn)
    scores = -loglik(test_bn, gF)  # baseline: higher = more real

    # Find EER threshold
    binary = (test_lbl == real_idx).astype(int)
    fpr, tpr, thrs = roc_curve(binary, scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    thr = thrs[eer_idx]
    preds = (scores >= thr).astype(int)  # 1=real, 0=fake

    print(f"EER threshold: {thr:.2f}")
    print(f"Overall: AUC={roc_auc_score(binary, scores):.4f}, "
          f"Acc={np.mean(preds==binary)*100:.2f}%")

    # Build dataframe with results
    wavenames = [os.path.basename(p).replace(".wav", "") for p in paths]
    gt_labels = np.array([inv[i] for i in test_lbl])

    df = pd.DataFrame({
        "wavename_base": wavenames,
        "gt_label": gt_labels,
        "is_real_gt": binary,
        "pred": preds,
        "score": scores,
        "correct": (preds == binary).astype(int),
    })

    # Merge with metadata
    df = df.merge(meta, on="wavename_base", how="left")
    matched = df["generator"].notna().sum()
    print(f"Matched {matched}/{len(df)} samples with metadata")

    # ── Error by GENERATOR ──
    print(f"\n{'='*80}")
    print("  Error Rate by Generator")
    print(f"{'='*80}")
    print(f"  {'Generator':15s} {'Count':>6s} {'Correct':>8s} {'Wrong':>7s} {'Acc%':>7s} {'ErrRate':>8s}")
    print("  " + "-" * 55)
    for gen in sorted(df["generator"].dropna().unique()):
        mask = df["generator"] == gen
        n = mask.sum()
        correct = df.loc[mask, "correct"].sum()
        wrong = n - correct
        acc = correct / n * 100
        print(f"  {gen:15s} {n:6d} {correct:8d} {wrong:7d} {acc:7.2f}% {100-acc:7.2f}%")

    # ── Error by SOURCE DATASET ──
    print(f"\n{'='*80}")
    print("  Error Rate by Source Dataset")
    print(f"{'='*80}")
    print(f"  {'Dataset':20s} {'Count':>6s} {'Correct':>8s} {'Wrong':>7s} {'Acc%':>7s} {'ErrRate':>8s}")
    print("  " + "-" * 60)
    for ds in sorted(df["source dataset"].dropna().unique()):
        mask = df["source dataset"] == ds
        n = mask.sum()
        correct = df.loc[mask, "correct"].sum()
        wrong = n - correct
        acc = correct / n * 100
        print(f"  {ds:20s} {n:6d} {correct:8d} {wrong:7d} {acc:7.2f}% {100-acc:7.2f}%")

    # ── Error by GENERATOR x SOURCE DATASET ──
    print(f"\n{'='*80}")
    print("  Error Rate by Generator x Source Dataset")
    print(f"{'='*80}")
    print(f"  {'Generator':15s} {'Dataset':20s} {'Count':>6s} {'Acc%':>7s} {'ErrRate':>8s}")
    print("  " + "-" * 65)
    grouped = df.dropna(subset=["generator"]).groupby(["generator", "source dataset"])
    for (gen, ds), grp in sorted(grouped, key=lambda x: (x[0][0], -len(x[1]))):
        n = len(grp)
        acc = grp["correct"].mean() * 100
        if n >= 10:  # skip tiny groups
            print(f"  {gen:15s} {ds:20s} {n:6d} {acc:7.2f}% {100-acc:7.2f}%")

    # ── Worst combinations (highest error rate, min 50 samples) ──
    print(f"\n{'='*80}")
    print("  TOP 15 Worst Generator x Dataset (min 50 samples)")
    print(f"{'='*80}")
    results = []
    for (gen, ds), grp in grouped:
        n = len(grp)
        if n >= 50:
            acc = grp["correct"].mean() * 100
            results.append((gen, ds, n, acc))
    results.sort(key=lambda x: x[3])  # sort by acc ascending
    print(f"  {'Generator':15s} {'Dataset':20s} {'Count':>6s} {'Acc%':>7s} {'ErrRate':>8s}")
    print("  " + "-" * 65)
    for gen, ds, n, acc in results[:15]:
        print(f"  {gen:15s} {ds:20s} {n:6d} {acc:7.2f}% {100-acc:7.2f}%")

    # ── Real samples: error by dataset ──
    print(f"\n{'='*80}")
    print("  Real Samples — Error by Source Dataset")
    print(f"{'='*80}")
    real_df = df[df["is_real_gt"] == 1]
    print(f"  {'Dataset':20s} {'Count':>6s} {'Correct':>8s} {'Wrong':>7s} {'Acc%':>7s}")
    print("  " + "-" * 50)
    for ds in sorted(real_df["source dataset"].dropna().unique()):
        mask = real_df["source dataset"] == ds
        n = mask.sum()
        correct = real_df.loc[mask, "correct"].sum()
        acc = correct / n * 100
        print(f"  {ds:20s} {n:6d} {correct:8d} {n-correct:7d} {acc:7.2f}%")

    print("\nDone!")


if __name__ == "__main__":
    main()
