"""
Retrain the 5-class model with higher real-class weights.

Uses the ORIGINAL model_beat architecture (Lightning-compatible),
loads from existing checkpoint, and fine-tunes with configurable CCL real_w.

Output checkpoints are saved in Lightning-compatible format so they work
with inference_all.py / eval_new_ckpt.py / load_pipeline_and_ccl.

Usage:
    python scripts/retrain_realw.py --ccl_real_w 8 --ce_w 3 --ce_real_w 8 --epochs 8
"""
import os, sys, logging, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import Namespace
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.linalg import cho_factor, cho_solve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from base_dataset import BeatsDataset, BalancedGeneratorSampler
from beats.model_beat import model_beat
from utils.training_utils import CenterContrastiveLoss, ArcFaceLoss, CenterLoss


# ── Eval helpers ──────────────────────────────────────────────────────────────
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


def compute_eer(labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2.0, thr[idx]


def eval_score(score, labels):
    auc_pos = roc_auc_score(labels, score)
    auc_neg = roc_auc_score(labels, -score)
    if auc_neg > auc_pos:
        score = -score
        auc = auc_neg
    else:
        auc = auc_pos
    eer, thr = compute_eer(labels, score)
    preds = (score >= thr).astype(int)
    rm = labels == 1
    fm = labels == 0
    real_acc = float((preds[rm] == 1).mean()) if rm.any() else 0
    fake_acc = float((preds[fm] == 0).mean()) if fm.any() else 0
    return {"auc": auc, "eer": eer, "real_acc": real_acc, "fake_acc": fake_acc}


# ── Data helpers ──────────────────────────────────────────────────────────────
SAMPLES_PER_LABEL_5 = {
    "fake_ata_01": 6, "fake_tta_01": 6, "fake_tta_02": 6, "fake_tta_03": 6, "real": 24,
}


def make_train_dl(json_file, batch_size=48):
    ds_args = Namespace(num_label=5, three_loss=True, audio_aug=False,
                        audio_mixup=False, audio_aug_prob=0, audio_mixup_prob=0)
    ds = BeatsDataset(json_file=json_file, transformation=None, args=ds_args)
    sampler = BalancedGeneratorSampler(ds, SAMPLES_PER_LABEL_5)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=4, pin_memory=True)
    return ds, dl


def make_eval_dl(json_file, batch_size=512):
    ds_args = Namespace(num_label=5, three_loss=True, audio_aug=False,
                        audio_mixup=False, audio_aug_prob=0, audio_mixup_prob=0)
    ds = BeatsDataset(json_file=json_file, transformation=None, args=ds_args)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True)
    return ds, dl


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def load_checkpoint(ckpt_path, device):
    """Load existing Lightning checkpoint into model_beat + CCL + ArcFace."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Build model_beat
    pipeline = model_beat(num_label=5, three_loss=True, feature_layer="predictor").to(device)

    # Extract pipeline weights (strip "pipeline." prefix)
    pipe_state = {}
    for k, v in state_dict.items():
        if k.startswith("pipeline."):
            pipe_state[k[9:]] = v

    # Load with shape check (flexible for head size changes)
    cur = pipeline.state_dict()
    compat = {k: v for k, v in pipe_state.items() if k in cur and v.shape == cur[k].shape}
    pipeline.load_state_dict(compat, strict=False)
    log.info("Pipeline: loaded %d/%d keys", len(compat), len(cur))

    # CCL head
    ccl_head = CenterContrastiveLoss(embed_dim=527, n_classes=2, m=0.7, s=30, lambda_c=2)
    ccl_state = {}
    for k, v in state_dict.items():
        if k.startswith("center_constrastive_loss."):
            ccl_state[k.replace("center_constrastive_loss.", "")] = v
    if ccl_state:
        ccl_head.load_state_dict(ccl_state, strict=False)
        log.info("CCL head: loaded %d keys", len(ccl_state))
    ccl_head = ccl_head.to(device)

    # ArcFace head
    arc_head = ArcFaceLoss(embed_dim=527, n_classes=5, m=3, s=30)
    arc_state = {}
    for k, v in state_dict.items():
        if k.startswith("arcface_loss."):
            arc_state[k.replace("arcface_loss.", "")] = v
    if arc_state:
        arc_head.load_state_dict(arc_state, strict=False)
        log.info("ArcFace head: loaded %d keys", len(arc_state))
    arc_head = arc_head.to(device)

    # CenterLoss head (fresh — no pretrained weights)
    center_head = CenterLoss().to(device)

    return pipeline, ccl_head, arc_head, center_head


def save_lightning_compat(pipeline, ccl_head, arc_head, filepath):
    """Save in Lightning-compatible format (state_dict key with prefixed names)."""
    state_dict = {}
    for k, v in pipeline.state_dict().items():
        state_dict[f"pipeline.{k}"] = v
    for k, v in ccl_head.state_dict().items():
        state_dict[f"center_constrastive_loss.{k}"] = v
    for k, v in arc_head.state_dict().items():
        state_dict[f"arcface_loss.{k}"] = v
    torch.save({"state_dict": state_dict}, filepath)
    log.info("Saved: %s", filepath)


# ── Collect features for LL eval ──────────────────────────────────────────────
@torch.no_grad()
def collect_features(pipeline, json_file, device):
    ds_args = Namespace(num_label=5, three_loss=True, audio_aug=False,
                        audio_mixup=False, audio_aug_prob=0, audio_mixup_prob=0)
    ds = BeatsDataset(json_file=json_file, transformation=None, args=ds_args)
    inv_label = {v: k for k, v in ds.label.items()}
    dl = DataLoader(ds, batch_size=512, num_workers=4, shuffle=False, pin_memory=True)
    all_bn, all_sm, all_lbl = [], [], []
    pipeline.eval()
    for batch in dl:
        audio = batch["audio"].to(device, dtype=torch.float32)
        outputs = pipeline.forward_pipeline(audio)
        all_bn.append(outputs[0].float().cpu().numpy())
        all_sm.append(outputs[1].float().cpu().numpy())
        if "label" in batch:
            all_lbl.extend(batch["label"].cpu().numpy())
        del audio, outputs
    return np.concatenate(all_bn), np.concatenate(all_sm), np.array(all_lbl), inv_label


# ── Evaluate on test sets ─────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_all(pipeline, ccl_head, ref_fake, ref_real, test_sets, device, epoch_label=""):
    """Evaluate baseline, LR, and CCL p(real) on all test sets."""
    pipeline.eval()
    ccl_head.eval()

    # Fit Gaussians on reference sets
    F_bn, _, _, _ = collect_features(pipeline, ref_fake, device)
    R_bn, _, _, _ = collect_features(pipeline, ref_real, device)
    gF = fit_gaussian(F_bn)
    gR = fit_gaussian(R_bn)

    results = {}
    for ds_name, ds_path in test_sets.items():
        T_bn, T_sm, T_lbl, inv = collect_features(pipeline, ds_path, device)
        real_idx = {v: k for k, v in inv.items()}["real"]
        labels = (T_lbl == real_idx).astype(int)

        # Baseline: -LL(fake)
        s1 = -loglik(T_bn, gF)
        r1 = eval_score(s1, labels)

        # LR: LL(real) - LL(fake)
        s2 = loglik(T_bn, gR) - loglik(T_bn, gF)
        r2 = eval_score(s2, labels)

        # Softmax p(real)
        p_real = T_sm[:, 4]
        r3 = eval_score(p_real, labels)

        results[ds_name] = {"baseline": r1, "LR": r2, "softmax": r3}

    # Print results
    print(f"\n  {epoch_label}")
    print(f"  {'Test Set':20s} {'Method':20s} {'AUC':>7s} {'EER':>8s} {'Real%':>8s} {'Fake%':>8s}")
    print("  " + "-" * 70)
    for ds_name, methods in results.items():
        for i, (method, r) in enumerate(methods.items()):
            prefix = ds_name if i == 0 else ""
            print(f"  {prefix:20s} {method:20s} {r['auc']:7.4f} {r['eer']*100:7.2f}% "
                  f"{r['real_acc']*100:7.2f}% {r['fake_acc']*100:7.2f}%")

    pipeline.train()
    ccl_head.train()
    return results


# ── Training ──────────────────────────────────────────────────────────────────
def train_one_epoch(pipeline, ccl_head, arc_head, center_head, train_dl, optimizer,
                    device, args, epoch):
    pipeline.train()
    ccl_head.train()
    arc_head.train()

    total_loss = 0.0
    n_batches = 0
    n_nan = 0
    n_correct_real = 0
    n_real = 0
    n_correct_fake = 0
    n_fake = 0

    for batch_idx, batch in enumerate(train_dl):
        audio = batch["audio"].to(device, dtype=torch.float32)
        label = batch["label"].to(device)

        # Always compute in float32 for numerical stability
        # (ArcFace s=30 → exp(30) overflows float16)
        bonafide_head, softmax_head, contrastive_head = pipeline.forward_pipeline(audio)
        bonafide_head = bonafide_head.float()
        softmax_head = softmax_head.float()

        label_5class = label.long()
        label_2class = (label == 4).long()  # real=4 in 5-class

        # ArcFace loss (5-class angular margin)
        _, arcface_loss = arc_head.forward(x=bonafide_head, y=label_5class)

        # CCL loss (2-class with configurable real_w)
        ccl_loss, ccl_logits = ccl_head.forward(x=bonafide_head, labels=label_2class)

        if args.no_ccl:
            loss = args.arc_w * arcface_loss
        else:
            loss = args.arc_w * arcface_loss + (1.0 - args.arc_w) * ccl_loss

        # Optional: CenterLoss on real class
        if args.center_w > 0:
            label_onehot = F.one_hot(label_5class, num_classes=5).float()
            cl = center_head.forward(bonafide_head, label_onehot, id_=4)  # real=4
            loss = loss + args.center_w * cl

        # Optional: weighted cross-entropy on softmax head
        # NOTE: softmax_head is ALREADY softmax'd (model_beat has nn.Softmax in head)
        # → use nll_loss(log(probs)) instead of cross_entropy(logits)
        if args.ce_w > 0:
            ce_class_w = torch.tensor(
                [1.0, 1.0, 1.0, 1.0, args.ce_real_w], device=device
            )
            log_probs = torch.log(softmax_head.clamp(min=1e-12))
            ce_loss = F.nll_loss(log_probs, label_5class, weight=ce_class_w)
            loss = loss + args.ce_w * ce_loss

        # NaN guard
        if torch.isnan(loss) or torch.isinf(loss):
            log.warning("NaN/Inf loss at batch %d — skipping", batch_idx)
            optimizer.zero_grad()
            n_nan += 1
            del audio, label, bonafide_head, softmax_head, contrastive_head
            continue

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            list(pipeline.parameters()) + list(ccl_head.parameters()) +
            list(arc_head.parameters()) + list(center_head.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # Track per-class accuracy from CCL logits
        with torch.no_grad():
            ccl_preds = ccl_logits.argmax(dim=1)  # 0=fake, 1=real
            real_mask = label_2class == 1
            fake_mask = label_2class == 0
            n_correct_real += (ccl_preds[real_mask] == 1).sum().item()
            n_real += real_mask.sum().item()
            n_correct_fake += (ccl_preds[fake_mask] == 0).sum().item()
            n_fake += fake_mask.sum().item()

        del audio, label, bonafide_head, softmax_head, contrastive_head

    avg_loss = total_loss / max(n_batches, 1)
    real_acc = n_correct_real / max(n_real, 1)
    fake_acc = n_correct_fake / max(n_fake, 1)
    bal_acc = (real_acc + fake_acc) / 2.0

    log.info("[Epoch %d] loss=%.4f  real=%.4f  fake=%.4f  bal=%.4f  (%d batches, %d NaN skipped)",
             epoch, avg_loss, real_acc, fake_acc, bal_acc, n_batches, n_nan)
    return avg_loss, real_acc, fake_acc, bal_acc


def main():
    parser = argparse.ArgumentParser(description="Retrain with higher real-class weights")
    parser.add_argument("--ckpt", type=str,
                        default="checkpoint/Beats_journal/Beats_Event_2stage_scratch_stage2_LR1.5e-7_ce3_realW4_8epoch/sample-03.ckpt",
                        help="Source checkpoint to fine-tune from")
    parser.add_argument("--train_json", type=str,
                        default="data/label/beats/Event_train_stage1.json",
                        help="Training data JSON (5-class)")
    parser.add_argument("--ref_fake", type=str,
                        default="data/label/beats/Event_train_stage1_fakeonly.json")
    parser.add_argument("--ref_real", type=str,
                        default="data/label/beats/Event_train_stage1_realonly.json")

    parser.add_argument("--ccl_real_w", type=float, default=8.0,
                        help="Real-class weight in CCL loss")
    parser.add_argument("--ce_w", type=float, default=3.0,
                        help="Weight for cross-entropy loss (0=disabled)")
    parser.add_argument("--ce_real_w", type=float, default=8.0,
                        help="Real-class weight in cross-entropy")
    parser.add_argument("--lr", type=float, default=5e-8,
                        help="Learning rate")
    parser.add_argument("--head_lr_mult", type=float, default=10.0,
                        help="LR multiplier for loss heads (CCL, ArcFace)")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--freeze_beats", action="store_true", default=False,
                        help="Freeze BEATs encoder (only train heads)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Override save directory")
    parser.add_argument("--arc_w", type=float, default=0.2,
                        help="ArcFace loss weight (default 0.2, CCL weight = 1 - arc_w)")
    parser.add_argument("--center_w", type=float, default=0.0,
                        help="CenterLoss weight on real class (0=disabled)")
    parser.add_argument("--no_ccl", action="store_true", default=False,
                        help="Disable CCL loss entirely")
    parser.add_argument("--keep_top_k", type=int, default=3,
                        help="Keep top-K checkpoints by balanced accuracy")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Derive save directory from args
    if args.save_dir is None:
        center_tag = f"_centerW{args.center_w}" if args.center_w > 0 else ""
        ccl_tag = "noCCL" if args.no_ccl else f"cclW{args.ccl_real_w:.0f}"
        args.save_dir = (
            f"checkpoint/Beats_journal/"
            f"retrain_{ccl_tag}_ceW{args.ce_w:.0f}_ceRW{args.ce_real_w:.0f}"
            f"{center_tag}_LR{args.lr:.0e}_{args.epochs}ep"
        )
    os.makedirs(args.save_dir, exist_ok=True)
    log.info("Save dir: %s", args.save_dir)

    # Test sets
    test_sets = {
        "test_track2": "data/label/beats/test_track2.json",
        "Event_test": "data/label/beats/old/audio_labels_beats_event_test.json",
        "TUTASC19_test": "data/label/beats/TUTASC19_test.json",
    }

    # Load checkpoint
    log.info("Loading checkpoint: %s", args.ckpt)
    pipeline, ccl_head, arc_head, center_head = load_checkpoint(args.ckpt, device)

    # Update CCL real_w
    ccl_head.real_w = args.ccl_real_w
    log.info("CCL real_w set to %.1f", ccl_head.real_w)

    # Optionally freeze BEATs backbone
    if args.freeze_beats:
        for name, param in pipeline.named_parameters():
            if name.startswith("BEATs."):
                param.requires_grad = False
        n_frozen = sum(1 for p in pipeline.parameters() if not p.requires_grad)
        n_total = sum(1 for p in pipeline.parameters())
        log.info("Frozen %d/%d pipeline params (BEATs backbone)", n_frozen, n_total)

    # Build optimizer with param groups
    pipeline_params = [p for p in pipeline.parameters() if p.requires_grad]
    head_params = list(ccl_head.parameters()) + list(arc_head.parameters()) + list(center_head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": pipeline_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=1e-2, betas=(0.9, 0.999))

    # Build train dataloader
    log.info("Loading training data: %s", args.train_json)
    train_ds, train_dl = make_train_dl(args.train_json)
    log.info("Training samples: %d", len(train_ds))

    # Evaluate baseline (before any training)
    log.info("Evaluating baseline (before training)...")
    evaluate_all(pipeline, ccl_head, args.ref_fake, args.ref_real, test_sets, device,
                 "BASELINE (before training)")

    # Train
    saved_ckpts = []  # (bal_acc, path) — sorted ascending
    for epoch in range(args.epochs):
        loss, real_acc, fake_acc, bal_acc = train_one_epoch(
            pipeline, ccl_head, arc_head, center_head, train_dl, optimizer,
            device, args, epoch)

        # Save checkpoint
        ckpt_name = f"epoch-{epoch:02d}_bal{bal_acc:.4f}.ckpt"
        ckpt_path = os.path.join(args.save_dir, ckpt_name)
        save_lightning_compat(pipeline, ccl_head, arc_head, ckpt_path)
        saved_ckpts.append((bal_acc, ckpt_path))

        # Keep only top-K
        if len(saved_ckpts) > args.keep_top_k:
            saved_ckpts.sort(key=lambda x: x[0])
            worst_acc, worst_path = saved_ckpts.pop(0)
            if os.path.exists(worst_path):
                os.remove(worst_path)
                log.info("Removed: %s (bal=%.4f)", worst_path, worst_acc)

        # Evaluate every 2 epochs or on last epoch
        if (epoch + 1) % 2 == 0 or epoch == args.epochs - 1:
            evaluate_all(pipeline, ccl_head, args.ref_fake, args.ref_real, test_sets, device,
                         f"Epoch {epoch} (cclW={args.ccl_real_w}, ceW={args.ce_w}, ceRW={args.ce_real_w})")

    log.info("Done! Checkpoints in: %s", args.save_dir)
    # Print final summary of saved checkpoints
    saved_ckpts.sort(key=lambda x: -x[0])
    print("\nSaved checkpoints:")
    for bal_acc, path in saved_ckpts:
        print(f"  bal={bal_acc:.4f}  {os.path.basename(path)}")


if __name__ == "__main__":
    main()
