"""
Batch t-SNE: 3 test sets × 2 checkpoints = 6 plots.
Extracts 527-dim backbone embeddings, runs t-SNE, saves PNG.
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

LABEL_COLORS_2 = {"real": "#2ecc71", "fake": "#e74c3c"}
LABEL_COLORS_5 = {
    "real": "#2ecc71", "fake_ata_01": "#e74c3c",
    "fake_tta_01": "#3498db", "fake_tta_02": "#f39c12", "fake_tta_03": "#9b59b6",
}


def load_pipeline(ckpt_path, prev_num_label, device):
    pipeline = model_beat(num_label=prev_num_label, three_loss=True, feature_layer="predictor")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    pipe_sd = {}
    for k, v in state.items():
        if k.startswith("pipeline."):
            pipe_sd[k[len("pipeline."):]] = v
    if not pipe_sd:
        pipe_sd = state
    pipeline.load_state_dict(pipe_sd, strict=False)
    pipeline.to(device).eval()
    return pipeline


def extract_embeddings(pipeline, json_file, device, max_samples=2000):
    with open(json_file) as f:
        data = json.load(f)

    by_label = defaultdict(list)
    for item in data:
        by_label[item["label"]].append(item)

    n_classes = len(by_label)
    per_class = max(100, max_samples // n_classes)

    sampled = []
    np.random.seed(42)
    for label, items in by_label.items():
        idx = np.random.choice(len(items), min(per_class, len(items)), replace=False)
        for i in idx:
            sampled.append(items[i])

    ds_args = argparse.Namespace(mode="beats", num_label=n_classes, get_first_dim=False)
    ds = BeatsDataset(json_file=json_file, transformation=None, args=ds_args)
    ds.data = sampled
    label_map = ds.label
    inv_label = {v: k for k, v in label_map.items()}

    dl = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=4, shuffle=False)

    all_embs, all_labels = [], []
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
    return embs, label_names


def plot_tsne(embs_2d, label_names, title, output_path):
    unique_labels = sorted(set(label_names))
    colors = LABEL_COLORS_5 if len(unique_labels) > 2 else LABEL_COLORS_2

    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl in unique_labels:
        mask = np.array([ln == lbl for ln in label_names])
        display = lbl.replace("_", " ").title()
        ax.scatter(
            embs_2d[mask, 0], embs_2d[mask, 1],
            c=colors.get(lbl, "#888888"), label=display,
            alpha=0.6, s=12, edgecolors="none",
        )

    ax.legend(fontsize=11, markerscale=3, loc="best")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoints = {
        "DIN-CTS": {
            "path": "checkpoint/BEATs_CTS/stage2/checkpoint/best.pt",
            "prev_num_label": 5,
        },
        "LL_fake_only": {
            "path": "checkpoint/Beats_journal/Beats_Event_2stage_scratch_stage2_LR1.5e-7_ce3_realW4_8epoch/sample-03.ckpt",
            "prev_num_label": 5,
        },
    }

    test_sets = {
        "test_track2": "data/label/beats/test_track2.json",
        "TUTASC19_test": "data/label/beats/TUTASC19_test.json",
        "GAM_event_test": "data/label/beats/old/audio_labels_beats_event_test.json",
    }

    out_dir = "inference_outputs/tsne"

    for ckpt_name, ckpt_info in checkpoints.items():
        print(f"\n=== Loading {ckpt_name} ===")
        pipeline = load_pipeline(ckpt_info["path"], ckpt_info["prev_num_label"], device)

        for ds_name, ds_path in test_sets.items():
            print(f"\n  Extracting embeddings: {ds_name}")
            embs, label_names = extract_embeddings(pipeline, ds_path, device, max_samples=2000)
            print(f"  Shape: {embs.shape}")

            print(f"  Running t-SNE...")
            tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                        max_iter=1000, learning_rate="auto", init="pca")
            embs_2d = tsne.fit_transform(embs)

            title = f"t-SNE 527-dim Embeddings — {ckpt_name} — {ds_name}"
            fname = f"tsne_{ckpt_name}_{ds_name}.png".replace(" ", "_")
            plot_tsne(embs_2d, label_names, title, os.path.join(out_dir, fname))

        del pipeline
        torch.cuda.empty_cache()

    print("\nAll done!")


if __name__ == "__main__":
    main()
