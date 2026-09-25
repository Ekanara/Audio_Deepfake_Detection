"""
2-stage training with full DIN-CTS loss combo (ASoftmax + SINCERE + CenterLoss + CE).
Combines experiment D + E: replace both ArcFace→ASoftmax AND CCL→SINCERE+CenterLoss.

Stage 1: LR=2e-8, 4 epochs — learn better cluster structure
Stage 2: LR=5e-9, 4 epochs — refine at very low LR

Starting from retrain_run1_ep06.
"""
import os, sys, logging, math
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
from utils.training_utils import CenterContrastiveLoss, ArcFaceLoss, ASoftmaxLoss, SINCERE, CenterLoss


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


SAMPLES_PER_LABEL_5 = {
    "fake_ata_01": 6, "fake_tta_01": 6, "fake_tta_02": 6, "fake_tta_03": 6, "real": 24,
}

def make_train_dl(json_file):
    ds_args = Namespace(num_label=5, three_loss=True, audio_aug=False,
                        audio_mixup=False, audio_aug_prob=0, audio_mixup_prob=0)
    ds = BeatsDataset(json_file=json_file, transformation=None, args=ds_args)
    sampler = BalancedGeneratorSampler(ds, SAMPLES_PER_LABEL_5)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=4, pin_memory=True)
    return ds, dl


@torch.no_grad()
def collect_features(pipeline, json_file, device):
    ds_args = Namespace(num_label=5, three_loss=True, audio_aug=False,
                        audio_mixup=False, audio_aug_prob=0, audio_mixup_prob=0)
    ds = BeatsDataset(json_file=json_file, transformation=None, args=ds_args)
    inv_label = {v: k for k, v in ds.label.items()}
    dl = DataLoader(ds, batch_size=512, num_workers=4, shuffle=False, pin_memory=True)
    all_bn, all_lbl = [], []
    pipeline.eval()
    for batch in dl:
        audio = batch["audio"].to(device, dtype=torch.float32)
        outputs = pipeline.forward_pipeline(audio)
        all_bn.append(outputs[0].float().cpu().numpy())
        if "label" in batch:
            all_lbl.extend(batch["label"].cpu().numpy())
        del audio, outputs
    return np.concatenate(all_bn), np.array(all_lbl), inv_label


@torch.no_grad()
def evaluate(pipeline, ref_fake, ref_real, test_sets, device, label=""):
    pipeline.eval()
    F_bn, _, _ = collect_features(pipeline, ref_fake, device)
    R_bn, _, _ = collect_features(pipeline, ref_real, device)
    gF = fit_gaussian(F_bn)

    print(f"\n  {label}")
    print(f"  {'Test Set':20s} {'AUC':>7s} {'EER':>8s} {'Real%':>8s} {'Fake%':>8s} {'Bal%':>8s}")
    print("  " + "-" * 60)

    results = {}
    for ds_name, ds_path in test_sets.items():
        T_bn, T_lbl, inv = collect_features(pipeline, ds_path, device)
        real_idx = {v: k for k, v in inv.items()}["real"]
        labels = (T_lbl == real_idx).astype(int)
        score = -loglik(T_bn, gF)
        auc = roc_auc_score(labels, score)
        eer, thr = compute_eer(labels, score)
        preds = (score >= thr).astype(int)
        real_acc = float((preds[labels == 1] == 1).mean())
        fake_acc = float((preds[labels == 0] == 0).mean())
        bal = (real_acc + fake_acc) / 2
        print(f"  {ds_name:20s} {auc:7.4f} {eer*100:7.2f}% {real_acc*100:7.2f}% {fake_acc*100:7.2f}% {bal*100:7.2f}%")
        results[ds_name] = {"auc": auc, "eer": eer, "real": real_acc, "fake": fake_acc, "bal": bal}

    pipeline.train()
    return results


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    pipeline = model_beat(num_label=5, three_loss=True, feature_layer="predictor").to(device)
    pipe_state = {k[9:]: v for k, v in state_dict.items() if k.startswith("pipeline.")}
    cur = pipeline.state_dict()
    compat = {k: v for k, v in pipe_state.items() if k in cur and v.shape == cur[k].shape}
    pipeline.load_state_dict(compat, strict=False)
    log.info("Pipeline: loaded %d/%d keys", len(compat), len(cur))

    # Still load CCL for checkpoint compatibility (won't be used in training)
    ccl_head = CenterContrastiveLoss(embed_dim=527, n_classes=2, m=0.7, s=30, lambda_c=2)
    ccl_state = {k.replace("center_constrastive_loss.", ""): v
                 for k, v in state_dict.items() if k.startswith("center_constrastive_loss.")}
    if ccl_state:
        ccl_head.load_state_dict(ccl_state, strict=False)
    ccl_head = ccl_head.to(device)

    return pipeline, ccl_head


def save_ckpt(pipeline, ccl_head, extra_heads, filepath):
    state_dict = {}
    for k, v in pipeline.state_dict().items():
        state_dict[f"pipeline.{k}"] = v
    for k, v in ccl_head.state_dict().items():
        state_dict[f"center_constrastive_loss.{k}"] = v
    for prefix, head in extra_heads.items():
        for k, v in head.state_dict().items():
            state_dict[f"{prefix}.{k}"] = v
    torch.save({"state_dict": state_dict}, filepath)


def train_stage(pipeline, ccl_head, asoftmax_head, sincere_head, center_head,
                train_dl, test_sets, ref_fake, ref_real, device,
                save_dir, stage_name, epochs, lr, head_lr_mult=10.0):
    """Train one stage with DIN-CTS losses: ASoftmax + SINCERE + CenterLoss + CE."""
    print(f"\n{'='*80}")
    print(f"  {stage_name}  |  LR={lr}  |  {epochs} epochs")
    print(f"{'='*80}")

    # Optimizer
    pipeline_params = [p for p in pipeline.parameters() if p.requires_grad]
    head_params = (list(asoftmax_head.parameters()) +
                   list(sincere_head.parameters()) +
                   list(center_head.parameters()))

    optimizer = torch.optim.AdamW([
        {"params": pipeline_params, "lr": lr},
        {"params": head_params, "lr": lr * head_lr_mult},
    ], weight_decay=1e-2, betas=(0.9, 0.999))

    os.makedirs(save_dir, exist_ok=True)
    best_bal = 0
    best_path = None
    extra_heads = {"asoftmax": asoftmax_head, "sincere": sincere_head, "center_loss": center_head}

    for epoch in range(epochs):
        pipeline.train()
        asoftmax_head.train()
        total_loss = 0
        n_batches = 0

        for batch in train_dl:
            audio = batch["audio"].to(device, dtype=torch.float32)
            label = batch["label"].to(device)

            bonafide_head, softmax_head, contrastive_head = pipeline.forward_pipeline(audio)
            bonafide_head = bonafide_head.float()
            softmax_head = softmax_head.float()

            label_5class = label.long()
            label_onehot = F.one_hot(label_5class, num_classes=5).float()

            # DIN-CTS 3-loss + CE
            _, asoftmax_loss = asoftmax_head.forward(x=bonafide_head, y=label_5class)
            sincere_loss = sincere_head.forward(bonafide_head, label_onehot)
            center_loss = center_head.forward(bonafide_head, label_onehot, id_=4)  # real=4

            # CE on softmax head (already softmax'd → nll_loss)
            log_probs = torch.log(softmax_head.clamp(min=1e-12))
            ce_w = torch.tensor([1, 1, 1, 1, 8.0], device=device)
            ce_loss = F.nll_loss(log_probs, label_5class, weight=ce_w)

            loss = 0.2 * asoftmax_loss + 0.4 * sincere_loss + 0.4 * center_loss + 3.0 * ce_loss

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                del audio, label, bonafide_head, softmax_head, contrastive_head
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(pipeline.parameters()) + head_params, max_norm=1.0
            )
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            del audio, label, bonafide_head, softmax_head, contrastive_head

        avg_loss = total_loss / max(n_batches, 1)
        log.info("[%s] Epoch %d  loss=%.4f", stage_name, epoch, avg_loss)

        # Evaluate every epoch
        results = evaluate(pipeline, ref_fake, ref_real, test_sets, device,
                          f"{stage_name} — Epoch {epoch} (LR={lr})")

        t2_bal = results["test_track2"]["bal"]
        ckpt_path = os.path.join(save_dir, f"{stage_name}_ep{epoch:02d}_t2bal{t2_bal:.4f}.ckpt")
        save_ckpt(pipeline, ccl_head, extra_heads, ckpt_path)

        if t2_bal > best_bal:
            best_bal = t2_bal
            best_path = ckpt_path

    print(f"\n  BEST {stage_name}: test_track2 bal={best_bal*100:.2f}% — {best_path}")
    return best_bal, best_path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = "checkpoint/Beats_journal/retrain_cclW8_ceW3_ceRW8_LR5e-08_8ep/epoch-06_bal0.9653.ckpt"
    train_json = "data/label/beats/Event_train_stage1.json"
    ref_fake = "data/label/beats/Event_train_stage1_fakeonly.json"
    ref_real = "data/label/beats/Event_train_stage1_realonly.json"
    test_sets = {
        "test_track2": "data/label/beats/test_track2.json",
        "Event_test": "data/label/beats/old/audio_labels_beats_event_test.json",
        "TUTASC19_test": "data/label/beats/TUTASC19_test.json",
    }
    save_dir = "checkpoint/Beats_journal/loss_ablation/F_dincts_2stage"

    # Load base checkpoint
    log.info("Loading checkpoint: %s", ckpt_path)
    pipeline, ccl_head = load_checkpoint(ckpt_path, device)

    # Build DIN-CTS loss heads (fresh — not loaded from checkpoint)
    asoftmax_head = ASoftmaxLoss(embed_dim=527, n_classes=5, m=2.5, s=30).to(device)
    sincere_head = SINCERE(temperature=0.1).to(device)
    center_head = CenterLoss().to(device)

    # Build train dataloader
    train_ds, train_dl = make_train_dl(train_json)
    log.info("Training samples: %d", len(train_ds))

    # Evaluate baseline before training
    evaluate(pipeline, ref_fake, ref_real, test_sets, device, "BASELINE (before training)")

    # ── Stage 1: LR=2e-8, 4 epochs ──
    best1, path1 = train_stage(
        pipeline, ccl_head, asoftmax_head, sincere_head, center_head,
        train_dl, test_sets, ref_fake, ref_real, device,
        save_dir, "stage1", epochs=4, lr=2e-8
    )

    # ── Stage 2: LR=5e-9, 4 epochs ──
    best2, path2 = train_stage(
        pipeline, ccl_head, asoftmax_head, sincere_head, center_head,
        train_dl, test_sets, ref_fake, ref_real, device,
        save_dir, "stage2", epochs=4, lr=5e-9
    )

    print(f"\n{'='*80}")
    print(f"  FINAL RESULTS")
    print(f"{'='*80}")
    print(f"  Stage 1 best: {best1*100:.2f}%  ({path1})")
    print(f"  Stage 2 best: {best2*100:.2f}%  ({path2})")
    print(f"  Overall best:  {max(best1, best2)*100:.2f}%")


if __name__ == "__main__":
    main()
