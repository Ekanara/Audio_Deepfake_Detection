"""
Score-level ensemble evaluation.

Loads multiple checkpoints, extracts bonafide_head features from each,
computes LL-based scores independently, then averages scores before evaluation.

This can combine complementary checkpoints (e.g. one good at test_track2,
another good at Event_test/TUTASC19) without any additional training.

Usage:
    python scripts/ensemble_eval.py \
        --ckpts ckpt_A.ckpt ckpt_B.ckpt \
        --weights 0.5 0.5
"""
import os, sys, argparse, logging
import numpy as np
import torch
from argparse import Namespace
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from base_dataset import BeatsDataset
from beats.model_beat import model_beat
from utils.training_utils import CenterContrastiveLoss, ArcFaceLoss


# ── Eval helpers ─────────────────────────────────────────────────────────────
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


# ── Model loading ────────────────────────────────────────────────────────────
def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    pipeline = model_beat(num_label=5, three_loss=True, feature_layer="predictor").to(device)
    pipe_state = {}
    for k, v in state_dict.items():
        if k.startswith("pipeline."):
            pipe_state[k[9:]] = v
    cur = pipeline.state_dict()
    compat = {k: v for k, v in pipe_state.items() if k in cur and v.shape == cur[k].shape}
    pipeline.load_state_dict(compat, strict=False)
    log.info("  Pipeline: loaded %d/%d keys from %s", len(compat), len(cur),
             os.path.basename(ckpt_path))
    return pipeline


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


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Score-level ensemble evaluation")
    parser.add_argument("--ckpts", type=str, nargs="+", required=True,
                        help="Checkpoint paths to ensemble")
    parser.add_argument("--weights", type=float, nargs="+", default=None,
                        help="Ensemble weights (default: equal)")
    parser.add_argument("--ref_fake", type=str,
                        default="data/label/beats/Event_train_stage1_fakeonly.json")
    parser.add_argument("--ref_real", type=str,
                        default="data/label/beats/Event_train_stage1_realonly.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_models = len(args.ckpts)
    if args.weights is None:
        weights = [1.0 / n_models] * n_models
    else:
        assert len(args.weights) == n_models, "Must provide same number of weights as checkpoints"
        total = sum(args.weights)
        weights = [w / total for w in args.weights]

    test_sets = {
        "test_track2": "data/label/beats/test_track2.json",
        "Event_test": "data/label/beats/old/audio_labels_beats_event_test.json",
        "TUTASC19_test": "data/label/beats/TUTASC19_test.json",
    }

    print(f"\n  Ensemble of {n_models} checkpoints (weights={[f'{w:.2f}' for w in weights]}):")
    for i, ckpt in enumerate(args.ckpts):
        print(f"    [{i}] w={weights[i]:.2f}  {os.path.basename(ckpt)}")

    # For each checkpoint: load model, collect ref features, fit Gaussians,
    # compute per-test-set LL scores
    all_model_scores = {ds: {"baseline": [], "LR": [], "softmax": []} for ds in test_sets}
    test_labels = {}

    for i, ckpt_path in enumerate(args.ckpts):
        log.info("Loading model %d: %s", i, ckpt_path)
        pipeline = load_checkpoint(ckpt_path, device)

        # Fit Gaussians on reference sets
        log.info("  Collecting reference features...")
        F_bn, _, _, _ = collect_features(pipeline, args.ref_fake, device)
        R_bn, _, _, _ = collect_features(pipeline, args.ref_real, device)
        gF = fit_gaussian(F_bn)
        gR = fit_gaussian(R_bn)

        for ds_name, ds_path in test_sets.items():
            log.info("  Evaluating %s...", ds_name)
            T_bn, T_sm, T_lbl, inv = collect_features(pipeline, ds_path, device)
            real_idx = {v: k for k, v in inv.items()}["real"]
            labels = (T_lbl == real_idx).astype(int)
            test_labels[ds_name] = labels

            # Baseline: -LL(fake)
            s_baseline = -loglik(T_bn, gF)
            # LR: LL(real) - LL(fake)
            s_lr = loglik(T_bn, gR) - loglik(T_bn, gF)
            # Softmax: p(real)
            s_softmax = T_sm[:, 4]

            all_model_scores[ds_name]["baseline"].append(s_baseline)
            all_model_scores[ds_name]["LR"].append(s_lr)
            all_model_scores[ds_name]["softmax"].append(s_softmax)

        # Free GPU memory
        del pipeline
        torch.cuda.empty_cache()

    # Now compute ensemble scores and evaluate
    # Also evaluate individual models for comparison
    print(f"\n  === INDIVIDUAL MODEL RESULTS ===")
    for i in range(n_models):
        print(f"\n  Model {i}: {os.path.basename(args.ckpts[i])}")
        print(f"  {'Test Set':20s} {'Method':20s} {'AUC':>7s} {'EER':>8s} {'Real%':>8s} {'Fake%':>8s}")
        print("  " + "-" * 70)
        for ds_name in test_sets:
            labels = test_labels[ds_name]
            for j, method in enumerate(["baseline", "LR", "softmax"]):
                scores = all_model_scores[ds_name][method][i]
                r = eval_score(scores, labels)
                prefix = ds_name if j == 0 else ""
                print(f"  {prefix:20s} {method:20s} {r['auc']:7.4f} {r['eer']*100:7.2f}% "
                      f"{r['real_acc']*100:7.2f}% {r['fake_acc']*100:7.2f}%")

    print(f"\n  === ENSEMBLE RESULTS (weighted average of scores) ===")
    print(f"  {'Test Set':20s} {'Method':20s} {'AUC':>7s} {'EER':>8s} {'Real%':>8s} {'Fake%':>8s}")
    print("  " + "-" * 70)
    for ds_name in test_sets:
        labels = test_labels[ds_name]
        for j, method in enumerate(["baseline", "LR", "softmax"]):
            # Weighted average of scores
            ensemble_score = np.zeros_like(all_model_scores[ds_name][method][0])
            for i in range(n_models):
                ensemble_score += weights[i] * all_model_scores[ds_name][method][i]
            r = eval_score(ensemble_score, labels)
            prefix = ds_name if j == 0 else ""
            print(f"  {prefix:20s} {method:20s} {r['auc']:7.4f} {r['eer']*100:7.2f}% "
                  f"{r['real_acc']*100:7.2f}% {r['fake_acc']*100:7.2f}%")

    # Also try different weight combinations for the 2-model case
    if n_models == 2:
        print(f"\n  === WEIGHT SWEEP (model 0 weight) ===")
        best_per_ds = {}
        for w0 in [0.3, 0.4, 0.5, 0.6, 0.7]:
            w1 = 1.0 - w0
            for ds_name in test_sets:
                labels = test_labels[ds_name]
                s = w0 * all_model_scores[ds_name]["baseline"][0] + \
                    w1 * all_model_scores[ds_name]["baseline"][1]
                r = eval_score(s, labels)
                key = (ds_name, w0)
                if ds_name not in best_per_ds or r["real_acc"] > best_per_ds[ds_name][1]:
                    best_per_ds[ds_name] = (w0, r["real_acc"], r["fake_acc"], r["auc"], r["eer"])

        print(f"  {'Test Set':20s} {'Best w0':>8s} {'Real%':>8s} {'Fake%':>8s} {'AUC':>7s} {'EER':>8s}")
        print("  " + "-" * 60)
        for ds_name in test_sets:
            w0, ra, fa, auc, eer = best_per_ds[ds_name]
            print(f"  {ds_name:20s} {w0:8.1f} {ra*100:7.2f}% {fa*100:7.2f}% {auc:7.4f} {eer*100:7.2f}%")


if __name__ == "__main__":
    main()
