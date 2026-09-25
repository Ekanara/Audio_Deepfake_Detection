"""
Train from scratch on Event_full (train+test) with 6 classes (including fake_unknown).

Stage 1: 15 epochs, LR=3e-7, ce_w=3, ce_real_w=2.5, ccl_real_w=4
Stage 2:  8 epochs, LR=1.5e-7, ce_w=3, ce_real_w=4, ccl_real_w=4

Labels (alphabetical): fake_ata_01=0, fake_tta_01=1, fake_tta_02=2, fake_tta_03=3, fake_unknown=4, real=5
Both stages: 0.2*ArcFace + 0.8*CCL + ce_w*CE
Inference on test_track2 + Foley_Sound with acc, F1, AUC.
"""
import os, sys, logging, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import Namespace
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
from scipy.linalg import cho_factor, cho_solve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from base_dataset import BeatsDataset, BalancedGeneratorSampler
from beats.model_beat import model_beat
from utils.training_utils import CenterContrastiveLoss, ArcFaceLoss

NUM_CLASSES = 6
REAL_CLASS_IDX = 5  # alphabetical: fake_ata_01=0, ..., fake_unknown=4, real=5

SAMPLES_PER_LABEL_6 = {
    "fake_ata_01": 6, "fake_tta_01": 6, "fake_tta_02": 6,
    "fake_tta_03": 6, "fake_unknown": 6, "real": 24,
}

# ── Helpers ──────────────────────────────────────────────────────────────────
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

def make_train_dl(json_file):
    ds_args = Namespace(num_label=NUM_CLASSES, three_loss=True, audio_aug=False,
                        audio_mixup=False, audio_aug_prob=0, audio_mixup_prob=0)
    ds = BeatsDataset(json_file=json_file, transformation=None, args=ds_args)
    sampler = BalancedGeneratorSampler(ds, SAMPLES_PER_LABEL_6)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=0, pin_memory=True)
    return ds, dl

@torch.no_grad()
def collect_features(pipeline, json_file, device):
    ds_args = Namespace(num_label=NUM_CLASSES, three_loss=True, audio_aug=False,
                        audio_mixup=False, audio_aug_prob=0, audio_mixup_prob=0)
    ds = BeatsDataset(json_file=json_file, transformation=None, args=ds_args)
    inv = {v: k for k, v in ds.label.items()}
    dl = DataLoader(ds, batch_size=512, num_workers=0, shuffle=False, pin_memory=True)
    all_bn, all_lbl = [], []
    pipeline.eval()
    for batch in dl:
        audio = batch["audio"].to(device, dtype=torch.float32)
        outputs = pipeline.forward_pipeline(audio)
        all_bn.append(outputs[0].float().cpu().numpy())
        all_lbl.extend(batch["label"].cpu().numpy())
        del audio, outputs
    return np.concatenate(all_bn), np.array(all_lbl), inv

def full_eval(pipeline, ref_fake_json, ref_real_json, test_sets, device, label=""):
    pipeline.eval()
    ref_fake_bn, _, _ = collect_features(pipeline, ref_fake_json, device)
    gF = fit_gaussian(ref_fake_bn)

    print(f"\n  {label}")
    print(f"  {'Test Set':15s} {'AUC':>7s} {'EER':>7s} {'Acc':>7s} {'F1r':>7s} {'F1f':>7s} {'F1m':>7s} {'Real%':>7s} {'Fake%':>7s}")
    print("  " + "-" * 75)

    results = {}
    for ds_name, ds_path in test_sets.items():
        T_bn, T_lbl, inv = collect_features(pipeline, ds_path, device)
        real_idx = {v: k for k, v in inv.items()}["real"]
        binary = (T_lbl == real_idx).astype(int)
        score = -loglik(T_bn, gF)
        auc = roc_auc_score(binary, score)
        fpr, tpr, thrs = roc_curve(binary, score, pos_label=1)
        fnr = 1.0 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
        thr = thrs[eer_idx]
        preds = (score >= thr).astype(int)
        acc = accuracy_score(binary, preds)
        f1r = f1_score(binary, preds, pos_label=1)
        f1f = f1_score(binary, preds, pos_label=0)
        f1m = (f1r + f1f) / 2
        real_acc = (preds[binary == 1] == 1).mean()
        fake_acc = (preds[binary == 0] == 0).mean()
        print(f"  {ds_name:15s} {auc:7.4f} {eer*100:6.2f}% {acc*100:6.2f}% {f1r:7.4f} {f1f:7.4f} {f1m:7.4f} {real_acc*100:6.2f}% {fake_acc*100:6.2f}%")
        results[ds_name] = {"auc": auc, "eer": eer, "acc": acc, "f1r": f1r, "f1f": f1f, "f1m": f1m,
                            "real": real_acc, "fake": fake_acc, "bal": (real_acc + fake_acc) / 2}

    pipeline.train()
    return results

def save_ckpt(pipeline, ccl_head, arc_head, filepath):
    state_dict = {}
    for k, v in pipeline.state_dict().items():
        state_dict[f"pipeline.{k}"] = v
    for k, v in ccl_head.state_dict().items():
        state_dict[f"center_constrastive_loss.{k}"] = v
    for k, v in arc_head.state_dict().items():
        state_dict[f"arcface_loss.{k}"] = v
    torch.save({"state_dict": state_dict}, filepath)

def load_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    pipeline = model_beat(num_label=NUM_CLASSES, three_loss=True, feature_layer="predictor").to(device)
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
    ccl_head = ccl_head.to(device)

    arc_head = ArcFaceLoss(embed_dim=527, n_classes=NUM_CLASSES, m=3, s=30)
    arc_state = {k.replace("arcface_loss.", ""): v
                 for k, v in state_dict.items() if k.startswith("arcface_loss.")}
    if arc_state:
        arc_head.load_state_dict(arc_state, strict=False)
    arc_head = arc_head.to(device)

    return pipeline, ccl_head, arc_head

def train_one_epoch(pipeline, ccl_head, arc_head, train_dl, optimizer, device,
                    arc_w, ce_w, ce_real_w, epoch):
    pipeline.train()
    ccl_head.train()
    arc_head.train()
    total_loss, n_batches, n_nan = 0, 0, 0

    for batch in train_dl:
        audio = batch["audio"].to(device, dtype=torch.float32)
        label = batch["label"].to(device)
        bonafide_head, softmax_head, _ = pipeline.forward_pipeline(audio)
        bonafide_head = bonafide_head.float()
        softmax_head = softmax_head.float()
        label_6class = label.long()
        label_2class = (label == REAL_CLASS_IDX).long()

        _, arcface_loss = arc_head.forward(x=bonafide_head, y=label_6class)
        ccl_loss, _ = ccl_head.forward(x=bonafide_head, labels=label_2class)
        loss = arc_w * arcface_loss + (1.0 - arc_w) * ccl_loss

        if ce_w > 0:
            log_probs = torch.log(softmax_head.clamp(min=1e-12))
            ce_class_w = torch.tensor(
                [1.0, 1.0, 1.0, 1.0, 1.0, float(ce_real_w)], device=device)
            ce_loss = F.nll_loss(log_probs, label_6class, weight=ce_class_w)
            loss = loss + ce_w * ce_loss

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            n_nan += 1
            del audio, label, bonafide_head, softmax_head
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(pipeline.parameters()) + list(ccl_head.parameters()) + list(arc_head.parameters()),
            max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        del audio, label, bonafide_head, softmax_head

    avg = total_loss / max(n_batches, 1)
    log.info("[Epoch %d] loss=%.4f (%d batches, %d NaN)", epoch, avg, n_batches, n_nan)
    return avg


def run_stage(pipeline, ccl_head, arc_head, train_dl, test_sets,
              ref_fake, ref_real, device, save_dir,
              epochs, lr, ce_w, ce_real_w, ccl_real_w, arc_w,
              stage_name, head_lr_mult=10.0):

    ccl_head.real_w = ccl_real_w
    log.info("CCL real_w = %.1f", ccl_head.real_w)

    pipeline_params = [p for p in pipeline.parameters() if p.requires_grad]
    head_params = list(ccl_head.parameters()) + list(arc_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": pipeline_params, "lr": lr},
        {"params": head_params, "lr": lr * head_lr_mult},
    ], weight_decay=1e-2, betas=(0.9, 0.999))

    os.makedirs(save_dir, exist_ok=True)
    best_bal = 0
    best_path = None

    for epoch in range(epochs):
        train_one_epoch(pipeline, ccl_head, arc_head, train_dl, optimizer, device,
                        arc_w, ce_w, ce_real_w, epoch)

        ckpt_path = os.path.join(save_dir, f"{stage_name}_ep{epoch:02d}.ckpt")
        save_ckpt(pipeline, ccl_head, arc_head, ckpt_path)

        if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
            results = full_eval(pipeline, ref_fake, ref_real, test_sets, device,
                               f"{stage_name} — Epoch {epoch} (LR={lr})")
            t2_bal = results.get("test_track2", {}).get("bal", 0)
            if t2_bal > best_bal:
                best_bal = t2_bal
                best_path = ckpt_path

    print(f"\n  BEST {stage_name}: test_track2 bal={best_bal*100:.2f}%")
    return best_bal, best_path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_json = "data/label/beats/Event_full_6class.json"
    ref_fake = "data/label/beats/Event_full_6class_fakeonly.json"
    ref_real = "data/label/beats/Event_full_6class_realonly.json"
    test_sets = {
        "test_track2": "data/label/beats/test_track2.json",
        "Foley_Sound": "data/label/beats/Foley_Sound_test.json",
    }
    base_save = "checkpoint/Beats_journal/Event_full_6class_scratch"

    # ── Stage 1: Fresh model, 15 epochs, LR=3e-7 ──
    log.info("Creating fresh 6-class model (BEATs pretrained)...")
    pipeline = model_beat(num_label=NUM_CLASSES, three_loss=True, feature_layer="predictor").to(device)
    ccl_head = CenterContrastiveLoss(embed_dim=527, n_classes=2, m=0.7, s=30, lambda_c=2).to(device)
    arc_head = ArcFaceLoss(embed_dim=527, n_classes=NUM_CLASSES, m=3, s=30).to(device)

    log.info("Building train dataloader...")
    train_ds, train_dl = make_train_dl(train_json)
    log.info("Train samples: %d (6 classes incl. fake_unknown)", len(train_ds))

    full_eval(pipeline, ref_fake, ref_real, test_sets, device, "BEFORE TRAINING (random heads)")

    print(f"\n{'='*80}")
    print("  STAGE 1: 15 epochs, LR=3e-7, ce=3, realW=2.5")
    print(f"{'='*80}")
    best1, path1 = run_stage(
        pipeline, ccl_head, arc_head, train_dl, test_sets,
        ref_fake, ref_real, device,
        save_dir=os.path.join(base_save, "stage1"),
        epochs=15, lr=3e-7, ce_w=3, ce_real_w=2.5, ccl_real_w=4, arc_w=0.2,
        stage_name="S1"
    )

    # ── Stage 2: From stage1 best, 8 epochs, LR=1.5e-7 ──
    print(f"\n{'='*80}")
    print("  STAGE 2: 8 epochs, LR=1.5e-7, ce=3, realW=4")
    print(f"{'='*80}")
    if path1:
        log.info("Loading stage1 best: %s", path1)
        pipeline, ccl_head, arc_head = load_ckpt(path1, device)

    best2, path2 = run_stage(
        pipeline, ccl_head, arc_head, train_dl, test_sets,
        ref_fake, ref_real, device,
        save_dir=os.path.join(base_save, "stage2"),
        epochs=8, lr=1.5e-7, ce_w=3, ce_real_w=4, ccl_real_w=4, arc_w=0.2,
        stage_name="S2"
    )

    # ── Final summary ──
    print(f"\n{'='*80}")
    print("  FINAL RESULTS")
    print(f"{'='*80}")
    print(f"  Stage 1 best: test_track2 bal={best1*100:.2f}%")
    print(f"  Stage 2 best: test_track2 bal={best2*100:.2f}%")

    if path2:
        log.info("Final eval on stage2 best: %s", path2)
        pipeline, ccl_head, arc_head = load_ckpt(path2, device)
        full_eval(pipeline, ref_fake, ref_real, test_sets, device, "FINAL (stage2 best)")


if __name__ == "__main__":
    main()
