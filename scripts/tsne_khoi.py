"""Regenerate Khoi t-SNE plots with renamed generator labels.
Uses exact same logic as inference_all.py plot_tsne_6label."""
import os, sys, numpy as np, torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from argparse import Namespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from base_dataset import BeatsDataset
from inference.inference_ccl import load_pipeline_and_ccl

COLORS_6 = {
    "real": "#2ecc71", "fake_ata_01": "#e74c3c",
    "fake_tta_01": "#3498db", "fake_tta_02": "#f39c12",
    "fake_tta_03": "#9b59b6", "fake_unknown": "#7f8c8d",
}
NAMES_6 = {
    "real": "Real", "fake_ata_01": "ATA-Audioldm1",
    "fake_tta_01": "TTA-Audiogen", "fake_tta_02": "TTA-Audioldm1",
    "fake_tta_03": "TTA-Audioldm2", "fake_unknown": "Unknown Fake",
}
ORDER_6 = ["real", "fake_ata_01", "fake_tta_01", "fake_tta_02", "fake_tta_03", "fake_unknown"]

@torch.no_grad()
def collect_features(pipeline, json_file, device):
    ds_args = Namespace(num_label=5, three_loss=True, audio_aug=False,
                        audio_mixup=False, audio_aug_prob=0, audio_mixup_prob=0)
    ds = BeatsDataset(json_file=json_file, transformation=None, args=ds_args)
    inv_label = {v: k for k, v in ds.label.items()}
    dl = DataLoader(ds, batch_size=512, num_workers=0, shuffle=False, pin_memory=True)
    all_bn, all_lbl = [], []
    pipeline.eval()
    for batch in dl:
        audio = batch["audio"].to(device, dtype=torch.float32)
        outputs = pipeline.forward_pipeline(audio)
        all_bn.append(outputs[0].float().cpu().numpy())
        all_lbl.extend(batch["label"].cpu().numpy())
        del audio, outputs
    return np.concatenate(all_bn), np.array(all_lbl), inv_label

def plot_tsne_6label(embs, gt_names, output_path):
    unique = sorted(set(gt_names))
    max_per_class = 400
    np.random.seed(42)
    gt_arr = np.array(gt_names)
    idx_parts = []
    for lbl in unique:
        where = np.where(gt_arr == lbl)[0]
        if len(where) > max_per_class:
            where = np.random.choice(where, max_per_class, replace=False)
        idx_parts.append(where)
    idx = np.sort(np.concatenate(idx_parts))
    sub_embs = embs[idx]
    sub_names = gt_arr[idx]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                max_iter=1000, learning_rate="auto", init="pca")
    embs_2d = tsne.fit_transform(sub_embs)

    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl in ORDER_6:
        if lbl not in unique:
            continue
        m = sub_names == lbl
        display = NAMES_6.get(lbl, lbl.replace("_", " ").title())
        ax.scatter(embs_2d[m, 0], embs_2d[m, 1],
                   c=COLORS_6.get(lbl, "#888"), label=display,
                   alpha=0.6, s=12, edgecolors="none")
    ax.legend(fontsize=11, markerscale=3, loc="best")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = "checkpoint/Beats_journal/Beats_Event_2stage_scratch_stage2_LR1.5e-7_ce3_realW4_8epoch/sample-03.ckpt"

print("Loading LL_fake_only checkpoint...")
pipeline, _, _ = load_pipeline_and_ccl(
    checkpoint_path=ckpt, device=device, mode="beats",
    num_classes=5, embed_dim=527, truncate_layers=0, beats_feature="predictor")

out_dir = "inference_outputs/tsne/LL_fake_only"

test_sets_5class = {
    "Event_test": "data/label/beats/Event_test_5class.json",
    "TUTASC19_test": "data/label/beats/TUTASC19_test_5class.json",
}

for ds_name, ds_path in test_sets_5class.items():
    print(f"\n{ds_name}...")
    bn, lbl, inv = collect_features(pipeline, ds_path, device)
    gt_names = [inv[l] for l in lbl]
    plot_tsne_6label(bn, gt_names, os.path.join(out_dir, f"Khoi_{ds_name}_5class.png"))

print("\nDone!")
