"""5-class classification on Event_test: can it tell which generator?"""
import os, sys, numpy as np, torch
from sklearn.metrics import classification_report, confusion_matrix
from argparse import Namespace
from torch.utils.data import DataLoader
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from base_dataset import BeatsDataset
from beats.model_beat import model_beat

device = "cuda"
SOFTMAX_MAP = {0: "fake_ata", 1: "fake_tta1", 2: "fake_tta2", 3: "fake_tta3", 4: "real"}

def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    pipeline = model_beat(num_label=5, three_loss=True, feature_layer="predictor").to(device)
    pipe_state = {k[9:]: v for k, v in state_dict.items() if k.startswith("pipeline.")}
    cur = pipeline.state_dict()
    compat = {k: v for k, v in pipe_state.items() if k in cur and v.shape == cur[k].shape}
    pipeline.load_state_dict(compat, strict=False)
    return pipeline

@torch.no_grad()
def collect(pipeline, json_file):
    ds_args = Namespace(num_label=5, three_loss=True, audio_aug=False,
                        audio_mixup=False, audio_aug_prob=0, audio_mixup_prob=0)
    ds = BeatsDataset(json_file=json_file, transformation=None, args=ds_args)
    inv = {v: k for k, v in ds.label.items()}
    dl = DataLoader(ds, batch_size=512, num_workers=4, shuffle=False, pin_memory=True)
    all_sm, all_lbl = [], []
    pipeline.eval()
    for batch in dl:
        audio = batch["audio"].to(device, dtype=torch.float32)
        outputs = pipeline.forward_pipeline(audio)
        all_sm.append(outputs[1].float().cpu().numpy())
        all_lbl.extend(batch["label"].cpu().numpy())
        del audio, outputs
    return np.concatenate(all_sm), np.array(all_lbl), inv

event5_json = "data/label/beats/Event_test_5class.json"

checkpoints = {
    "LL_fake_only (original)": "checkpoint/Beats_journal/Beats_Event_2stage_scratch_stage2_LR1.5e-7_ce3_realW4_8epoch/sample-03.ckpt",
    "retrain_run1_ep06": "checkpoint/Beats_journal/retrain_cclW8_ceW3_ceRW8_LR5e-08_8ep/epoch-06_bal0.9653.ckpt",
}

for ckpt_name, ckpt_path in checkpoints.items():
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  {ckpt_name}")
    print(sep)

    pipeline = load_model(ckpt_path)
    sm, lbl_int, inv = collect(pipeline, event5_json)

    gt_str = np.array([inv[i] for i in lbl_int])

    # Softmax predictions
    pred_idx = sm.argmax(axis=1)
    pred_str = np.array([SOFTMAX_MAP[i] for i in pred_idx])

    # Map GT to match pred names
    gt_mapped = np.array([
        {"fake_ata_01": "fake_ata", "fake_tta_01": "fake_tta1",
         "fake_tta_02": "fake_tta2", "fake_tta_03": "fake_tta3",
         "real": "real"}.get(g, g)
        for g in gt_str
    ])

    # 5-class eval on known classes (exclude fake_unknown)
    known_mask = gt_str != "fake_unknown"
    gt_k = gt_mapped[known_mask]
    pred_k = pred_str[known_mask]

    labels_5 = ["fake_ata", "fake_tta1", "fake_tta2", "fake_tta3", "real"]
    print(f"\n  5-Class Classification (known classes, n={known_mask.sum()})")
    print(classification_report(gt_k, pred_k, labels=labels_5, digits=4, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(gt_k, pred_k, labels=labels_5)
    header = "  " + f"{'GT':>12s}" + "".join(f"{'p_'+l:>10s}" for l in labels_5)
    print(header)
    for i, row_name in enumerate(labels_5):
        row = "  " + f"{row_name:>12s}" + "".join(f"{cm[i,j]:10d}" for j in range(5))
        print(row)

    # fake_unknown distribution
    unk_mask = gt_str == "fake_unknown"
    if unk_mask.sum() > 0:
        unk_preds = pred_str[unk_mask]
        print(f"\n  fake_unknown ({unk_mask.sum()} samples) predicted as:")
        for cls, cnt in sorted(Counter(unk_preds).items(), key=lambda x: -x[1]):
            print(f"    {cls:12s}: {cnt:5d} ({cnt/unk_mask.sum()*100:.1f}%)")

    del pipeline
    torch.cuda.empty_cache()

print("\nDone!")
