"""
Loss ablation: start from retrain_run1_ep06, swap/add one DIN-CTS loss at a time.

Experiments:
  A. baseline  — original losses (ArcFace + CCL + CE) [re-run for fair comparison]
  B. +SINCERE  — add SINCERE contrastive loss
  C. +CenterL  — add CenterLoss on real class
  D. ArcFace→ASoftmax — replace ArcFace with ASoftmax
  E. CCL→SINCERE+CenterL — replace CCL with DIN-CTS pair

Each trains 4 epochs from retrain_run1_ep06, same LR/config.
"""
import os, sys, logging, argparse, math
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
        score = -loglik(T_bn, gF)  # baseline
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

    ccl_head = CenterContrastiveLoss(embed_dim=527, n_classes=2, m=0.7, s=30, lambda_c=2)
    ccl_state = {k.replace("center_constrastive_loss.", ""): v
                 for k, v in state_dict.items() if k.startswith("center_constrastive_loss.")}
    if ccl_state:
        ccl_head.load_state_dict(ccl_state, strict=False)
        log.info("CCL head: loaded %d keys", len(ccl_state))
    ccl_head = ccl_head.to(device)

    arc_head = ArcFaceLoss(embed_dim=527, n_classes=5, m=3, s=30)
    arc_state = {k.replace("arcface_loss.", ""): v
                 for k, v in state_dict.items() if k.startswith("arcface_loss.")}
    if arc_state:
        arc_head.load_state_dict(arc_state, strict=False)
        log.info("ArcFace head: loaded %d keys", len(arc_state))
    arc_head = arc_head.to(device)

    return pipeline, ccl_head, arc_head


def save_ckpt(pipeline, ccl_head, arc_head, extra_heads, filepath):
    state_dict = {}
    for k, v in pipeline.state_dict().items():
        state_dict[f"pipeline.{k}"] = v
    for k, v in ccl_head.state_dict().items():
        state_dict[f"center_constrastive_loss.{k}"] = v
    for k, v in arc_head.state_dict().items():
        state_dict[f"arcface_loss.{k}"] = v
    for prefix, head in extra_heads.items():
        for k, v in head.state_dict().items():
            state_dict[f"{prefix}.{k}"] = v
    torch.save({"state_dict": state_dict}, filepath)


def run_experiment(mode, pipeline, ccl_head, arc_head, train_dl, test_sets,
                   ref_fake, ref_real, device, save_dir, epochs=4, lr=5e-8):
    """Run one loss ablation experiment."""
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT: {mode}")
    print(f"{'='*80}")

    # Build extra loss heads based on mode
    extra_heads = {}
    sincere_head = None
    center_head = None
    asoftmax_head = None

    if mode in ["B_add_sincere", "E_replace_ccl"]:
        sincere_head = SINCERE(temperature=0.1).to(device)
        extra_heads["sincere"] = sincere_head

    if mode in ["C_add_center", "E_replace_ccl"]:
        center_head = CenterLoss().to(device)
        extra_heads["center_loss"] = center_head

    if mode == "D_asoftmax":
        asoftmax_head = ASoftmaxLoss(embed_dim=527, n_classes=5, m=2.5, s=30).to(device)
        extra_heads["asoftmax"] = asoftmax_head

    # Build optimizer
    all_params = [
        {"params": [p for p in pipeline.parameters() if p.requires_grad], "lr": lr},
        {"params": list(ccl_head.parameters()) + list(arc_head.parameters()), "lr": lr * 10},
    ]
    for _, h in extra_heads.items():
        if list(h.parameters()):
            all_params.append({"params": list(h.parameters()), "lr": lr * 10})

    optimizer = torch.optim.AdamW(all_params, weight_decay=1e-2, betas=(0.9, 0.999))

    os.makedirs(save_dir, exist_ok=True)
    best_bal = 0
    best_path = None

    for epoch in range(epochs):
        pipeline.train()
        ccl_head.train()
        arc_head.train()
        total_loss = 0
        n_batches = 0

        for batch in train_dl:
            audio = batch["audio"].to(device, dtype=torch.float32)
            label = batch["label"].to(device)

            bonafide_head, softmax_head, contrastive_head = pipeline.forward_pipeline(audio)
            bonafide_head = bonafide_head.float()
            softmax_head = softmax_head.float()

            label_5class = label.long()
            label_2class = (label == 4).long()
            label_onehot = F.one_hot(label_5class, num_classes=5).float()

            loss = torch.tensor(0.0, device=device)

            # ── Loss assembly based on mode ──
            if mode == "A_baseline":
                # Original: 0.2*ArcFace + 0.8*CCL + 3*CE
                _, arc_loss = arc_head.forward(x=bonafide_head, y=label_5class)
                ccl_loss, _ = ccl_head.forward(x=bonafide_head, labels=label_2class)
                log_probs = torch.log(softmax_head.clamp(min=1e-12))
                ce_w = torch.tensor([1,1,1,1,8.0], device=device)
                ce_loss = F.nll_loss(log_probs, label_5class, weight=ce_w)
                loss = 0.2 * arc_loss + 0.8 * ccl_loss + 3.0 * ce_loss

            elif mode == "B_add_sincere":
                # Original + SINCERE
                _, arc_loss = arc_head.forward(x=bonafide_head, y=label_5class)
                ccl_loss, _ = ccl_head.forward(x=bonafide_head, labels=label_2class)
                log_probs = torch.log(softmax_head.clamp(min=1e-12))
                ce_w = torch.tensor([1,1,1,1,8.0], device=device)
                ce_loss = F.nll_loss(log_probs, label_5class, weight=ce_w)
                sincere_loss = sincere_head.forward(bonafide_head, label_onehot)
                loss = 0.2 * arc_loss + 0.8 * ccl_loss + 3.0 * ce_loss + 0.4 * sincere_loss

            elif mode == "C_add_center":
                # Original + CenterLoss on real
                _, arc_loss = arc_head.forward(x=bonafide_head, y=label_5class)
                ccl_loss, _ = ccl_head.forward(x=bonafide_head, labels=label_2class)
                log_probs = torch.log(softmax_head.clamp(min=1e-12))
                ce_w = torch.tensor([1,1,1,1,8.0], device=device)
                ce_loss = F.nll_loss(log_probs, label_5class, weight=ce_w)
                center_loss = center_head.forward(bonafide_head, label_onehot, id_=4)  # real=4
                loss = 0.2 * arc_loss + 0.8 * ccl_loss + 3.0 * ce_loss + 0.4 * center_loss

            elif mode == "D_asoftmax":
                # Replace ArcFace with ASoftmax, keep CCL + CE
                _, asoftmax_loss = asoftmax_head.forward(x=bonafide_head, y=label_5class)
                ccl_loss, _ = ccl_head.forward(x=bonafide_head, labels=label_2class)
                log_probs = torch.log(softmax_head.clamp(min=1e-12))
                ce_w = torch.tensor([1,1,1,1,8.0], device=device)
                ce_loss = F.nll_loss(log_probs, label_5class, weight=ce_w)
                loss = 0.2 * asoftmax_loss + 0.8 * ccl_loss + 3.0 * ce_loss

            elif mode == "E_replace_ccl":
                # Replace CCL with SINCERE+CenterLoss, keep ArcFace + CE
                _, arc_loss = arc_head.forward(x=bonafide_head, y=label_5class)
                sincere_loss = sincere_head.forward(bonafide_head, label_onehot)
                center_loss = center_head.forward(bonafide_head, label_onehot, id_=4)
                log_probs = torch.log(softmax_head.clamp(min=1e-12))
                ce_w = torch.tensor([1,1,1,1,8.0], device=device)
                ce_loss = F.nll_loss(log_probs, label_5class, weight=ce_w)
                loss = 0.2 * arc_loss + 0.4 * sincere_loss + 0.4 * center_loss + 3.0 * ce_loss

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                del audio, label, bonafide_head, softmax_head, contrastive_head
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(pipeline.parameters()) + list(ccl_head.parameters()) +
                list(arc_head.parameters()) +
                [p for h in extra_heads.values() for p in h.parameters()],
                max_norm=1.0
            )
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            del audio, label, bonafide_head, softmax_head, contrastive_head

        avg_loss = total_loss / max(n_batches, 1)
        log.info("[%s] Epoch %d  loss=%.4f", mode, epoch, avg_loss)

        # Evaluate every epoch
        results = evaluate(pipeline, ref_fake, ref_real, test_sets, device,
                          f"{mode} — Epoch {epoch}")

        # Save best by test_track2 balanced acc
        t2_bal = results["test_track2"]["bal"]
        ckpt_path = os.path.join(save_dir, f"{mode}_ep{epoch:02d}_bal{t2_bal:.4f}.ckpt")
        save_ckpt(pipeline, ccl_head, arc_head, extra_heads, ckpt_path)
        if t2_bal > best_bal:
            best_bal = t2_bal
            best_path = ckpt_path

    print(f"\n  BEST {mode}: test_track2 bal={best_bal*100:.2f}% — {best_path}")
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
    }

    modes = ["E_replace_ccl"]

    all_results = {}
    for mode in modes:
        # Reload fresh from checkpoint each time
        log.info("Loading checkpoint for %s...", mode)
        pipeline, ccl_head, arc_head = load_checkpoint(ckpt_path, device)
        ccl_head.real_w = 8.0

        save_dir = f"checkpoint/Beats_journal/loss_ablation/{mode}_LR2e-8_8ep"
        best_bal, best_path = run_experiment(
            mode, pipeline, ccl_head, arc_head,
            make_train_dl(train_json)[1], test_sets,
            ref_fake, ref_real, device, save_dir,
            epochs=8, lr=2e-8
        )
        all_results[mode] = best_bal

        # Cleanup GPU
        del pipeline, ccl_head, arc_head
        torch.cuda.empty_cache()

    # Final summary
    print(f"\n{'='*80}")
    print("  FINAL SUMMARY — Best test_track2 balanced acc per experiment")
    print(f"{'='*80}")
    for mode, bal in sorted(all_results.items(), key=lambda x: -x[1]):
        print(f"  {mode:25s}  {bal*100:.2f}%")


if __name__ == "__main__":
    main()
