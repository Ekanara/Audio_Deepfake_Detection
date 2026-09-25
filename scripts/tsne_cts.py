"""
t-SNE visualization of 527-dim backbone embeddings (5-class).
Uses the latest CTS checkpoint to extract embeddings from Event_train_stage1.
"""

import os, sys, json, argparse
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from beats.model_beat import model_beat
from base_dataset import BeatsDataset


LABEL_COLORS = {
    "real":        "#2ecc71",
    "fake_ata_01": "#e74c3c",
    "fake_tta_01": "#3498db",
    "fake_tta_02": "#f39c12",
    "fake_tta_03": "#9b59b6",
}

LABEL_NAMES = {
    "real":        "Real",
    "fake_ata_01": "Fake ATA",
    "fake_tta_01": "Fake TTA-1",
    "fake_tta_02": "Fake TTA-2",
    "fake_tta_03": "Fake TTA-3",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--json_file", default="data/label/beats/Event_train_stage1.json")
    parser.add_argument("--prev_num_label", type=int, default=5)
    parser.add_argument("--samples_per_class", type=int, default=800)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--output", default="inference_outputs/tsne_cts_5class.png")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load model
    pipeline = model_beat(
        num_label=args.prev_num_label,
        three_loss=True,
        feature_layer="predictor",
    )
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    pipe_sd = {}
    for k, v in state.items():
        if k.startswith("pipeline."):
            pipe_sd[k[len("pipeline."):]] = v
    if not pipe_sd:
        pipe_sd = state
    pipeline.load_state_dict(pipe_sd, strict=False)
    pipeline.to(device).eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    # Load data
    with open(args.json_file) as f:
        data = json.load(f)

    # Subsample per class
    by_label = defaultdict(list)
    for item in data:
        by_label[item["label"]].append(item)

    sampled = []
    for label, items in by_label.items():
        np.random.seed(42)
        idx = np.random.choice(len(items), min(args.samples_per_class, len(items)), replace=False)
        for i in idx:
            sampled.append(items[i])
    print(f"Sampled {len(sampled)} items ({args.samples_per_class}/class)")

    # Build dataset + dataloader
    ds_args = argparse.Namespace(mode="beats", num_label=5, get_first_dim=False)
    ds = BeatsDataset(json_file=args.json_file, transformation=None, args=ds_args)
    # Override ds.data with sampled subset
    ds.data = sampled

    dl = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=4, shuffle=False)

    # Extract embeddings
    all_embs, all_labels = [], []
    label_map = ds.label  # e.g. {"fake_ata_01": 0, ...}
    inv_label = {v: k for k, v in label_map.items()}

    with torch.no_grad():
        for batch in dl:
            audio = batch["audio"].to(device, dtype=torch.float32)
            labels = batch["label"].cpu().numpy()
            outputs = pipeline.forward_pipeline(audio)
            bonafide_head = outputs[0] if isinstance(outputs, tuple) else outputs
            all_embs.append(bonafide_head.cpu().numpy())
            all_labels.append(labels)

    embs = np.concatenate(all_embs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    label_names = [inv_label[l] for l in labels]
    print(f"Embeddings shape: {embs.shape}")

    # t-SNE
    print(f"Running t-SNE (perplexity={args.perplexity})...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=42,
                max_iter=1000, learning_rate="auto", init="pca")
    embs_2d = tsne.fit_transform(embs)
    print("t-SNE done.")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    for label_key in ["real", "fake_ata_01", "fake_tta_01", "fake_tta_02", "fake_tta_03"]:
        mask = np.array([ln == label_key for ln in label_names])
        if not mask.any():
            continue
        ax.scatter(
            embs_2d[mask, 0], embs_2d[mask, 1],
            c=LABEL_COLORS[label_key],
            label=LABEL_NAMES[label_key],
            alpha=0.6, s=12, edgecolors="none",
        )

    ax.legend(fontsize=11, markerscale=3, loc="best")
    ax.set_title("t-SNE of 527-dim BEATs Embeddings (5-class, CTS Stage 2)", fontsize=13)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
