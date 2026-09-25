"""Evaluate both checkpoints on Foley_Sound test set."""
import os, sys, numpy as np, torch
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
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
    print(f"  Pipeline: loaded {len(compat)}/{len(cur)} keys")
    return pipeline

@torch.no_grad()
def collect(pipeline, json_file):
    ds_args = Namespace(num_label=5, three_loss=True, audio_aug=False,
                        audio_mixup=False, audio_aug_prob=0, audio_mixup_prob=0)
    ds = BeatsDataset(json_file=json_file, transformation=None, args=ds_args)
    inv = {v: k for k, v in ds.label.items()}
    dl = DataLoader(ds, batch_size=512, num_workers=4, shuffle=False, pin_memory=True)
    all_bn, all_lbl = [], []
    pipeline.eval()
    for batch in dl:
        audio = batch["audio"].to(device, dtype=torch.float32)
        outputs = pipeline.forward_pipeline(audio)
        all_bn.append(outputs[0].float().cpu().numpy())
        all_lbl.extend(batch["label"].cpu().numpy())
        del audio, outputs
    return np.concatenate(all_bn), np.array(all_lbl), inv

def eval_and_print(name, score, binary):
    auc = roc_auc_score(binary, score)
    fpr, tpr, thrs = roc_curve(binary, score, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    thr = thrs[eer_idx]
    preds = (score >= thr).astype(int)
    acc = accuracy_score(binary, preds)
    f1_r = f1_score(binary, preds, pos_label=1)
    f1_f = f1_score(binary, preds, pos_label=0)
    real_acc = (preds[binary == 1] == 1).mean()
    fake_acc = (preds[binary == 0] == 0).mean()
    print(f"  {name}:")
    print(f"    AUC={auc:.4f}  EER={eer*100:.2f}%  Acc={acc*100:.2f}%")
    print(f"    F1(real)={f1_r:.4f}  F1(fake)={f1_f:.4f}  F1(macro)={(f1_r+f1_f)/2:.4f}")
    print(f"    Real%={real_acc*100:.2f}%  Fake%={fake_acc*100:.2f}%  Bal%={(real_acc+fake_acc)/2*100:.2f}%")

checkpoints = {
    "LL_fake_only (original)": "checkpoint/Beats_journal/Beats_Event_2stage_scratch_stage2_LR1.5e-7_ce3_realW4_8epoch/sample-03.ckpt",
    "retrain_run1_ep06": "checkpoint/Beats_journal/retrain_cclW8_ceW3_ceRW8_LR5e-08_8ep/epoch-06_bal0.9653.ckpt",
}

test_json = "data/label/beats/Foley_Sound_test.json"

for ckpt_name, ckpt_path in checkpoints.items():
    print(f"\n{'='*80}")
    print(f"  {ckpt_name}")
    print(f"{'='*80}")

    pipeline = load_model(ckpt_path)

    ref_fake_bn, _, _ = collect(pipeline, "data/label/beats/Event_train_stage1_fakeonly.json")
    ref_real_bn, _, _ = collect(pipeline, "data/label/beats/Event_train_stage1_realonly.json")
    test_bn, test_lbl, inv = collect(pipeline, test_json)

    real_idx = {v: k for k, v in inv.items()}["real"]
    binary = (test_lbl == real_idx).astype(int)
    print(f"  Test: {len(binary)} samples ({binary.sum()} real, {(1-binary).sum()} fake)")

    gF = fit_gaussian(ref_fake_bn)
    gR = fit_gaussian(ref_real_bn)
    ll_fake = loglik(test_bn, gF)
    ll_real = loglik(test_bn, gR)

    eval_and_print("baseline (-LL_fake)", -ll_fake, binary)
    eval_and_print("LR (LL_real - LL_fake)", ll_real - ll_fake, binary)

    del pipeline
    torch.cuda.empty_cache()

print("\nDone!")
