import logging
import os
from datetime import datetime

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class PeriodicEvaluationCallback(Callback):
    """
    Evaluate fake/real performance every N epochs and save artifacts.
    """

    CLASS_NAMES = ["fake", "real"]

    def __init__(
        self,
        eval_dataloader,
        eval_every_n_epochs=10,
        output_dir="eval_outputs",
        real_class_idx=0,
        device="cuda",
    ):
        super().__init__()
        self.eval_dataloader = eval_dataloader
        self.eval_every_n_epochs = eval_every_n_epochs
        self.output_dir = output_dir
        self.real_class_idx = real_class_idx
        self.device = device
        os.makedirs(self.output_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.eval_every_n_epochs != 0:
            return

        logger.info("[PeriodicEval] Epoch %d - running fake/real evaluation", epoch + 1)
        pipeline = pl_module.pipeline
        pipeline.eval()

        results = self._run_inference(pipeline)
        if "labels" in results:
            metrics = self._compute_metrics(results)
            self._log_metrics(trainer, metrics, epoch)
            self._save_plots(metrics, results, epoch)
            self._save_csv(results, metrics, epoch)
            self._print_summary(metrics, epoch)

        pipeline.train()

    def _run_inference(self, pipeline):
        all_prob_real = []
        all_labels = []
        all_paths = []

        with torch.no_grad():
            for batch in self.eval_dataloader:
                inputs = batch["audio"].to(self.device, dtype=torch.float32)
                labels = batch.get("label")
                paths = batch.get("path", [])

                model_output = pipeline.forward_pipeline(inputs)
                probs_mc = self._to_multiclass_probs(model_output)
                prob_real = probs_mc[:, self.real_class_idx]

                all_prob_real.extend(prob_real.tolist())
                all_paths.extend(list(paths))

                if labels is not None:
                    if isinstance(labels, torch.Tensor):
                        all_labels.extend(labels.cpu().numpy().tolist())
                    else:
                        all_labels.extend(np.asarray(labels).tolist())

        prob_real_arr = np.asarray(all_prob_real)
        prob_fake_arr = 1.0 - prob_real_arr

        results = {
            "paths": np.asarray(all_paths),
            "probabilities": np.stack([prob_fake_arr, prob_real_arr], axis=1),
        }
        if len(all_labels) > 0:
            results["labels"] = np.asarray(all_labels)

        self._apply_threshold(results, threshold=0.5)
        return results

    def _to_multiclass_probs(self, model_output):
        if isinstance(model_output, tuple):
            if len(model_output) >= 2:
                softmax_head = model_output[1]
                if isinstance(softmax_head, torch.Tensor) and softmax_head.ndim == 2:
                    return softmax_head.detach().cpu().numpy()
            logits = model_output[0]
        else:
            logits = model_output

        if not isinstance(logits, torch.Tensor):
            raise TypeError("Model output must be a tensor or tuple of tensors.")
        return F.softmax(logits, dim=1).detach().cpu().numpy()

    def _apply_threshold(self, results, threshold=0.5):
        prob_real = results["probabilities"][:, 1]
        prob_fake = results["probabilities"][:, 0]
        predictions = (prob_real >= threshold).astype(int)
        confidence = np.where(predictions == 1, prob_real, prob_fake)

        results["predictions"] = predictions
        results["confidences"] = confidence
        results["threshold"] = threshold

    def _compute_eer(self, labels, prob_real):
        fpr, tpr, thresholds = roc_curve(labels, prob_real, pos_label=1)
        fnr = 1.0 - tpr
        idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2.0
        return eer, thresholds[idx], fpr, tpr, fnr

    def _compute_metrics(self, results):
        labels = results["labels"]
        predictions = results["predictions"]
        probs = results["probabilities"]
        confidences = results["confidences"]

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, labels=[0, 1], zero_division=0
        )
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average="macro", zero_division=0
        )
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted", zero_division=0
        )

        try:
            auc = roc_auc_score(labels, probs[:, 1])
        except ValueError:
            auc = float("nan")

        try:
            eer, eer_threshold, fpr, tpr, fnr = self._compute_eer(labels, probs[:, 1])
        except ValueError:
            eer = float("nan")
            eer_threshold = float("nan")
            fpr = np.array([])
            tpr = np.array([])
            fnr = np.array([])

        cm = confusion_matrix(labels, predictions, labels=[0, 1])
        report = classification_report(
            labels,
            predictions,
            target_names=self.CLASS_NAMES,
            zero_division=0,
        )

        real_conf = confidences[labels == 1]
        fake_conf = confidences[labels == 0]

        return {
            "accuracy": accuracy,
            "eer": eer,
            "eer_threshold": eer_threshold,
            "auc": auc,
            "fpr": fpr,
            "tpr": tpr,
            "fnr": fnr,
            "precision_macro": p_macro,
            "recall_macro": r_macro,
            "f1_macro": f1_macro,
            "precision_weighted": p_weighted,
            "recall_weighted": r_weighted,
            "f1_weighted": f1_weighted,
            "per_class_precision": precision,
            "per_class_recall": recall,
            "per_class_f1": f1,
            "per_class_support": support,
            "confusion_matrix": cm,
            "report": report,
            "avg_confidence": float(np.mean(confidences)),
            "avg_conf_real": float(np.mean(real_conf)) if len(real_conf) else 0.0,
            "avg_conf_fake": float(np.mean(fake_conf)) if len(fake_conf) else 0.0,
        }

    def _log_metrics(self, trainer, metrics, epoch):
        log_dict = {
            "eval/accuracy": metrics["accuracy"],
            "eval/eer": metrics["eer"],
            "eval/auc": metrics["auc"],
            "eval/f1_macro": metrics["f1_macro"],
            "eval/f1_fake": metrics["per_class_f1"][0],
            "eval/f1_real": metrics["per_class_f1"][1],
            "eval/recall_fake": metrics["per_class_recall"][0],
            "eval/recall_real": metrics["per_class_recall"][1],
            "eval/avg_conf_fake": metrics["avg_conf_fake"],
            "eval/avg_conf_real": metrics["avg_conf_real"],
        }

        if trainer.logger is not None:
            trainer.logger.log_metrics(log_dict, step=epoch)

    def _epoch_dir(self, epoch):
        path = os.path.join(self.output_dir, f"epoch_{epoch + 1:04d}")
        os.makedirs(path, exist_ok=True)
        return path

    def _save_plots(self, metrics, results, epoch):
        epoch_dir = self._epoch_dir(epoch)
        self._plot_confusion_matrix(metrics, epoch_dir, epoch)
        self._plot_roc_curve(metrics, epoch_dir, epoch)
        self._plot_eer_curve(metrics, results, epoch_dir, epoch)
        self._plot_per_class_metrics(metrics, epoch_dir, epoch)
        self._plot_confidence_distribution(results, epoch_dir, epoch)

    def _plot_confusion_matrix(self, metrics, output_dir, epoch):
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            metrics["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.CLASS_NAMES,
            yticklabels=self.CLASS_NAMES,
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix - Epoch {epoch + 1}")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
        plt.close(fig)

    def _plot_roc_curve(self, metrics, output_dir, epoch):
        if len(metrics["fpr"]) == 0 or len(metrics["tpr"]) == 0:
            return

        eer_idx = np.argmin(np.abs(metrics["fpr"] - metrics["fnr"]))
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(
            metrics["fpr"],
            metrics["tpr"],
            label=f"AUC = {metrics['auc']:.4f}",
            color="steelblue",
        )
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.scatter(
            [metrics["fpr"][eer_idx]],
            [metrics["tpr"][eer_idx]],
            color="red",
            zorder=5,
            label=f"EER = {metrics['eer'] * 100:.2f}%",
        )
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - Epoch {epoch + 1}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
        plt.close(fig)

    def _plot_eer_curve(self, metrics, results, output_dir, epoch):
        if "labels" not in results or len(metrics["fpr"]) == 0 or len(metrics["fnr"]) == 0:
            return

        _, _, thresholds = roc_curve(results["labels"], results["probabilities"][:, 1], pos_label=1)
        min_len = min(len(thresholds), len(metrics["fpr"]), len(metrics["fnr"]))
        thresholds = thresholds[:min_len]
        fpr = metrics["fpr"][:min_len]
        fnr = metrics["fnr"][:min_len]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(thresholds, fpr, label="FAR (False Acceptance Rate)", color="blue")
        ax.plot(thresholds, fnr, label="FRR (False Rejection Rate)", color="red")
        ax.axvline(
            x=metrics["eer_threshold"],
            color="green",
            linestyle="--",
            label=f"EER threshold = {metrics['eer_threshold']:.4f}",
        )
        ax.scatter(
            [metrics["eer_threshold"]],
            [metrics["eer"]],
            color="green",
            zorder=5,
            label=f"EER = {metrics['eer'] * 100:.2f}%",
        )
        ax.set_xlabel("Threshold P(real)")
        ax.set_ylabel("Error Rate")
        ax.set_title(f"FAR / FRR - EER Curve - Epoch {epoch + 1}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eer_curve.png"), dpi=150)
        plt.close(fig)

    def _plot_per_class_metrics(self, metrics, output_dir, epoch):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        entries = [
            (metrics["per_class_precision"], "Precision"),
            (metrics["per_class_recall"], "Recall"),
            (metrics["per_class_f1"], "F1-Score"),
        ]
        for ax, (values, title) in zip(axes, entries):
            bars = ax.bar(self.CLASS_NAMES, values, color=["tomato", "steelblue"])
            ax.set_title(f"{title} - Epoch {epoch + 1}")
            ax.set_ylim(0, 1)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{value:.3f}",
                    ha="center",
                    fontsize=9,
                )

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "per_class_metrics.png"), dpi=150)
        plt.close(fig)

    def _plot_confidence_distribution(self, results, output_dir, epoch):
        if "labels" not in results:
            return

        labels = results["labels"]
        prob_real = results["probabilities"][:, 1]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(prob_real[labels == 1], bins=40, alpha=0.6, color="steelblue", label="Real")
        ax.hist(prob_real[labels == 0], bins=40, alpha=0.6, color="tomato", label="Fake")
        ax.axvline(
            x=results["threshold"],
            color="black",
            linestyle="--",
            label=f"threshold = {results['threshold']:.2f}",
        )
        ax.set_xlabel("P(real)")
        ax.set_ylabel("Count")
        ax.set_title(f"P(real) Distribution - Epoch {epoch + 1}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confidence_distribution.png"), dpi=150)
        plt.close(fig)

    def _save_csv(self, results, metrics, epoch):
        epoch_dir = self._epoch_dir(epoch)
        labels = results["labels"]

        pd.DataFrame(
            {
                "path": results["paths"],
                "label": ["real" if label == 1 else "fake" for label in labels],
                "prediction": [
                    "real" if pred == 1 else "fake" for pred in results["predictions"]
                ],
                "prob_fake": results["probabilities"][:, 0],
                "prob_real": results["probabilities"][:, 1],
                "confidence": results["confidences"],
                "correct": (results["predictions"] == labels).astype(int),
                "threshold": results["threshold"],
                "epoch": epoch + 1,
            }
        ).to_csv(os.path.join(epoch_dir, "per_audio_results.csv"), index=False)

        summary_path = os.path.join(self.output_dir, "training_summary.csv")
        pd.DataFrame(
            [
                {
                    "epoch": epoch + 1,
                    "accuracy": metrics["accuracy"],
                    "eer": metrics["eer"],
                    "eer_threshold": metrics["eer_threshold"],
                    "auc": metrics["auc"],
                    "f1_macro": metrics["f1_macro"],
                    "f1_fake": metrics["per_class_f1"][0],
                    "f1_real": metrics["per_class_f1"][1],
                    "recall_fake": metrics["per_class_recall"][0],
                    "recall_real": metrics["per_class_recall"][1],
                    "avg_conf_fake": metrics["avg_conf_fake"],
                    "avg_conf_real": metrics["avg_conf_real"],
                }
            ]
        ).to_csv(
            summary_path,
            mode="a",
            header=not os.path.exists(summary_path),
            index=False,
        )
        logger.info("[PeriodicEval] Results saved to %s", epoch_dir)

    def _print_summary(self, metrics, epoch):
        print(f"\n{'=' * 52}")
        print(f"  Periodic Eval - Epoch {epoch + 1}  [fake=0 / real=1]")
        print(f"{'=' * 52}")
        print(f"  Accuracy         : {metrics['accuracy']:.4f}")
        print(
            f"  EER              : {metrics['eer'] * 100:.2f}%"
            f"  (thr={metrics['eer_threshold']:.4f})"
        )
        print(f"  AUC-ROC          : {metrics['auc']:.4f}")
        print(f"  F1  macro        : {metrics['f1_macro']:.4f}")
        print(
            f"  F1  fake / real  : {metrics['per_class_f1'][0]:.4f}"
            f" / {metrics['per_class_f1'][1]:.4f}"
        )
        print(
            f"  Recall fake/real : {metrics['per_class_recall'][0]:.4f}"
            f" / {metrics['per_class_recall'][1]:.4f}"
        )
        print(
            f"  Conf fake / real : {metrics['avg_conf_fake']:.4f}"
            f" / {metrics['avg_conf_real']:.4f}"
        )
        print(f"{'=' * 52}\n")


class TrainingMonitorCallback(Callback):
    """
    Monitor batch progression and diagnose skipped ranges.
    """

    def __init__(self, dataset, batch_size, log_dir="./logs"):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.total_samples = len(dataset)
        self.expected_steps = (self.total_samples + batch_size - 1) // batch_size

        self.epoch_steps = []
        self.last_step = -1
        self.skip_detected = False

        self.monitor_log = os.path.join(log_dir, "training_monitor.log")
        self.error_log = os.path.join(log_dir, "errors.log")

        with open(self.monitor_log, "w") as f:
            f.write(
                f"Training Monitor - Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Total samples: {self.total_samples}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Expected steps per epoch: {self.expected_steps}\n")
            f.write("=" * 80 + "\n\n")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        del pl_module, batch

        if self.last_step != -1 and batch_idx != self.last_step + 1:
            self.skip_detected = True
            skip_msg = f"\nSKIP DETECTED at epoch {trainer.current_epoch}"
            skip_msg += f"\nStep jumped from {self.last_step} to {batch_idx}"
            skip_msg += f"\nMissing steps: {list(range(self.last_step + 1, batch_idx))}"

            skip_start = (self.last_step + 1) * self.batch_size
            skip_end = batch_idx * self.batch_size
            skip_msg += f"\nAffected samples: {skip_start} to {skip_end - 1}"

            print(skip_msg)
            with open(self.monitor_log, "a") as f:
                f.write(f"{datetime.now().strftime('%H:%M:%S')} - {skip_msg}\n\n")

            self._diagnose_batch(skip_start, skip_end, trainer.current_epoch)

        self.last_step = batch_idx
        self.epoch_steps.append(batch_idx)

    def on_train_epoch_end(self, trainer, pl_module):
        del pl_module
        completed_steps = len(self.epoch_steps)

        summary = f"\n{'=' * 80}\n"
        summary += f"Epoch {trainer.current_epoch} training summary:\n"
        summary += f"Expected steps: {self.expected_steps}\n"
        summary += f"Completed steps: {completed_steps}\n"

        if completed_steps < self.expected_steps:
            missing = self.expected_steps - completed_steps
            summary += f"Missing {missing} steps\n"

            last_step = self.epoch_steps[-1] if self.epoch_steps else -1
            summary += f"Last completed step: {last_step}\n"

            skip_start = (last_step + 1) * self.batch_size
            skip_end = self.expected_steps * self.batch_size
            summary += f"Skipped samples: {skip_start} to {skip_end - 1}\n"

            print(summary)
            with open(self.monitor_log, "a") as f:
                f.write(f"{datetime.now().strftime('%H:%M:%S')} - {summary}\n")

            if not self.skip_detected:
                self._diagnose_batch(skip_start, skip_end, trainer.current_epoch)
        else:
            summary += "All steps completed\n"
            print(summary)
            with open(self.monitor_log, "a") as f:
                f.write(f"{datetime.now().strftime('%H:%M:%S')} - {summary}\n")

        summary += f"{'=' * 80}\n"

        self.epoch_steps = []
        self.last_step = -1
        self.skip_detected = False

    def _diagnose_batch(self, start_idx, end_idx, epoch):
        diagnosis_file = os.path.join(
            self.log_dir, f"diagnosis_epoch{epoch}_batch{start_idx}-{end_idx}.txt"
        )

        problematic = []
        target_fs = getattr(self.dataset, "fs", None)

        for idx in range(start_idx, min(end_idx, len(self.dataset.data))):
            item = self.dataset.data[idx]
            audio_path = item["audio"]
            label = item["label"]

            issues = []
            if not os.path.exists(audio_path):
                issues.append("File does not exist")
            else:
                try:
                    file_size = os.path.getsize(audio_path)
                    if file_size == 0:
                        issues.append("File is empty (0 bytes)")
                    elif file_size < 1000:
                        issues.append(f"File suspiciously small ({file_size} bytes)")

                    wav, sr = sf.read(audio_path)
                    if wav.ndim > 1:
                        wav = wav[:, 0]

                    if len(wav) == 0:
                        issues.append("Audio has no samples")
                    if np.isnan(wav).any():
                        issues.append("Contains NaN values")
                    if np.isinf(wav).any():
                        issues.append("Contains Inf values")
                    if wav.max() == wav.min():
                        issues.append("Audio is constant")

                    if target_fs is not None and sr != target_fs:
                        wav_resampled = librosa.core.resample(
                            wav, orig_sr=sr, target_sr=target_fs
                        )
                        if np.isnan(wav_resampled).any() or np.isinf(wav_resampled).any():
                            issues.append("NaN/Inf after resampling")
                except Exception as exc:
                    issues.append(f"Failed to inspect file: {exc}")

            if issues:
                problematic.append((idx, audio_path, label, issues))

        with open(diagnosis_file, "w") as f:
            f.write("Batch Diagnosis Report\n")
            f.write(f"Epoch: {epoch}\n")
            f.write(f"Batch range: {start_idx} to {end_idx - 1}\n")
            f.write(f"Diagnosed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Found {len(problematic)} problematic samples\n")
            f.write("=" * 80 + "\n\n")

            if problematic:
                for idx, path, label, issues in problematic:
                    f.write(f"Index: {idx}\n")
                    f.write(f"Path: {path}\n")
                    f.write(f"Label: {label}\n")
                    f.write("Issues:\n")
                    for issue in issues:
                        f.write(f"  - {issue}\n")
                    f.write("\n")
            else:
                f.write("No obvious file issues found.\n")
                f.write("Possible causes:\n")
                f.write("  - DataLoader worker crash\n")
                f.write("  - Memory issues\n")
                f.write("  - Transformation errors\n")
                f.write("  - Multiprocessing issues\n")

        with open(self.error_log, "a") as f:
            f.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Epoch {epoch}\n")
            f.write(f"Batch {start_idx}-{end_idx}: {len(problematic)} issues\n")
            for idx, path, _label, issues in problematic[:3]:
                f.write(f"  {idx}: {path} - {issues[0]}\n")
            f.write("\n")

        return problematic


def create_trainer(args):
    os.system(f"wandb login --relogin {args.wandb_api}")
    wandb_logger = WandbLogger(
        project=args.wandb_run_name,
        log_model=False,
    )

    model_checkpoint = ModelCheckpoint(
        save_top_k=args.save_top_k,
        monitor="valid/loss",
        mode="min",
        dirpath=args.output_dir,
        filename="sample-{epoch:02d}-{valid/loss:.2f}",
        save_weights_only=False,
    )

    dist = len(args.devices) > 1
    trainer = Trainer(
        max_epochs=args.num_train_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[model_checkpoint],
        strategy="ddp_find_unused_parameters_true" if dist else "auto",
        log_every_n_steps=args.log_steps,
        logger=wandb_logger,
        precision=args.precision,
        accumulate_grad_batches=args.gradient_accumulation_steps,
    )

    device_idx = trainer.global_rank if dist else 0
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    return trainer, device, dist


def save_checkpoint(trainer, args):
    save_ckpt_path = f"{args.save_ckpt_path}/checkpoint"
    os.makedirs(save_ckpt_path, exist_ok=True)
    trainer.save_checkpoint(f"{save_ckpt_path}/best.pt")


def load_from_disk_(dataset_name, split="train"):
    dataset_root = dataset_name.split("/")[-1]
    dataset_path = f"{dataset_root}/{split}"
    return load_from_disk(dataset_path)
