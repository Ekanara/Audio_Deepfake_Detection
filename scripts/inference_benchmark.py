import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from base_trainer import build_model
from base_pipeline import BasePipeline
from base_dataset import BaselineDataset


def compute_eer(labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2.0, thr[idx]


def load_checkpoint(mode, ckpt_dir, num_label=2, device="cuda"):
    model = build_model(mode)
    args = argparse.Namespace(mode=mode, num_label=num_label, get_first_dim=False)
    pipeline = BasePipeline(args=args, model=model, device=device, in_features=3, num_label=num_label, mode=mode)

    # Look for best.pt first (saved by our training scripts), then fall back to .ckpt
    ckpt_files = []
    for root, dirs, files in os.walk(ckpt_dir):
        for f in files:
            if f == "best.pt":
                ckpt_files.insert(0, os.path.join(root, f))  # prioritise best.pt
            elif f.endswith((".ckpt", ".pt")):
                ckpt_files.append(os.path.join(root, f))

    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")

    # best.pt is already first if it exists; otherwise pick most recent
    if not ckpt_files[0].endswith("best.pt"):
        ckpt_files.sort(key=os.path.getmtime, reverse=True)
    best_ckpt = ckpt_files[0]

    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    pipe_sd = {}
    for k, v in state.items():
        if k.startswith("pipeline."):
            pipe_sd[k[len("pipeline."):]] = v
    if not pipe_sd:
        pipe_sd = state

    pipeline.load_state_dict(pipe_sd, strict=False)
    pipeline.to(device).eval()
    print(f"Loaded: {best_ckpt}")
    return pipeline


@torch.no_grad()
def run_inference(pipeline, test_json, mode, device="cuda", batch_size=64):
    args = argparse.Namespace(mode=mode, num_label=2, get_first_dim=False)
    dataset = BaselineDataset(json_file=test_json, transformation=None, args=args)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    real_idx = dataset.label.get("real", 1)
    all_probs, all_labels = [], []

    for batch in loader:
        audio = batch["audio"].to(device, dtype=torch.float32)
        labels = batch["label"].cpu().numpy()
        logits = pipeline.forward_pipeline(audio)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append((labels == real_idx).astype(int))

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return probs, labels


def print_metrics(name, probs, labels):
    p_real = probs[:, 1]
    preds = probs.argmax(axis=1)

    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, p_real)
    eer, eer_thr = compute_eer(labels, p_real)
    f1 = f1_score(labels, preds, average="binary")

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.2f}")
    print(f"  F1:        {f1:.2f}")
    print(f"  AUC:       {auc:.2f}")
    print(f"  EER:       {eer:.2f}")
    return {"name": name, "acc": acc, "f1": f1, "auc": auc, "eer": eer}


PRESETS = {
    "tutasc19": {
        "VGG16":        ("vgg",       "checkpoint/VGG16/VGG16_Gam_TUTASC19_bench"),
        "MobileNetV2":  ("mobilenet", "checkpoint/MobileNetV2/MobileNetV2_Gam_TUTASC19_bench"),
        "ConvNeXt-Tiny":("convnext",  "checkpoint/ConvNeXt_Tiny/ConvNeXt_Tiny_Gam_TUTASC19_bench"),
        "Xception":     ("xception",  "checkpoint/Xception/Xception_Gam_TUTASC19_bench"),
        "NASNet-Large": ("nasnet",    "checkpoint/NASNet/NASNet_Gam_TUTASC19_bench"),
    },
    "gam_old": {
        "VGG16":        ("vgg",       "checkpoint/VGG16/VGG16_Gam_old_bench"),
        "MobileNetV2":  ("mobilenet", "checkpoint/MobileNetV2/MobileNetV2_Gam_old_bench"),
        "ConvNeXt-Tiny":("convnext",  "checkpoint/ConvNeXt_Tiny/ConvNeXt_Tiny_Gam_old_bench"),
        "Xception":     ("xception",  "checkpoint/Xception/Xception_Gam_old_bench"),
        "NASNet-Large": ("nasnet",    "checkpoint/NASNet/NASNet_Gam_old_bench"),
    },
}
MODELS = PRESETS["tutasc19"]  # default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_json", default="data/label/gam/TUTASC19_test.json")
    parser.add_argument("--preset", default=None, choices=list(PRESETS.keys()), help="Checkpoint preset (tutasc19 or gam_old)")
    parser.add_argument("--models", nargs="*", default=None, help="Model names to evaluate, default=all")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    models = PRESETS[args.preset] if args.preset else MODELS
    targets = args.models if args.models else list(models.keys())
    results = []

    for name in targets:
        if name not in models:
            print(f"Unknown model: {name}, skipping")
            continue
        mode, ckpt_dir = models[name]
        if not os.path.exists(ckpt_dir):
            print(f"Checkpoint dir not found: {ckpt_dir}, skipping {name}")
            continue
        try:
            pipeline = load_checkpoint(mode, ckpt_dir, device=device)
            probs, labels = run_inference(pipeline, args.test_json, mode, device, args.batch_size)
            r = print_metrics(name, probs, labels)
            results.append(r)
            del pipeline
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"ERROR {name}: {e}")

    if results:
        print(f"\n{'='*70}")
        print(f"  BENCHMARK SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Model':<16} {'Acc':>6} {'F1':>6} {'AUC':>6} {'EER':>6}")
        print(f"  {'-'*42}")
        for r in sorted(results, key=lambda x: -x["auc"]):
            print(f"  {r['name']:<16} {r['acc']:.2f}  {r['f1']:.2f}  {r['auc']:.2f}  {r['eer']:.2f}")


if __name__ == "__main__":
    main()
