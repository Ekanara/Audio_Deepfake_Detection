import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from utils.arguments import get_args
from base_pipeline import BasePipeline
from base_dataset import BaselineDataset
from torch.utils.data import DataLoader
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.models import efficientnet_b1, inception_v3
from torchvision import transforms
import timm
from utils.training_utils import apply_reshape, feature_extract


def load_pipeline(checkpoint_path=None, device='cuda', num_classes=2):
    """Load the pipeline directly from checkpoint"""
    logger.info(f"Loading pipeline from {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        backbone_model = efficientnet_b1(pretrained=False)
        
        pipeline = BasePipeline(
            model=backbone_model,
            args=None,
            device=device,
            in_features=2048,
            num_label=num_classes
        ).to(device)
        
        if 'state_dict' in checkpoint:
            pipeline_state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('pipeline.'):
                    # Remove 'pipeline.' prefix
                    new_key = key[9:]  # Remove 'pipeline.'
                    pipeline_state_dict[new_key] = value
            
            # Load the state dict
            pipeline.load_state_dict(pipeline_state_dict)
            logger.info("Pipeline loaded from Lightning checkpoint")
        else:
            # Direct pipeline checkpoint
            pipeline.load_state_dict(checkpoint)
            logger.info("Pipeline loaded from direct checkpoint")
            
        pipeline.eval()
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        raise
    
    return pipeline

def predict_single(pipeline, input_tensors, device):
    with torch.no_grad():
        if len(input_tensors.shape) == 3:
            input_tensors = input_tensors.unsqueeze(0)

        input_tensors = input_tensors.to(device)
        logits = pipeline.forward_pipeline(input_tensors)
        probability = F.softmax(input=logits, dim=1)
        predicted_class = torch.argmax(input = probability, dim=1)
        confidence = torch.max(input = probability, dim=1)[0]

    return {
        "logits": logits,
        "probability": probability,
        "predicted_class": predicted_class,
        "confidence": confidence
    }

def predicted_batch(pipeline, dataloader, device='cuda', feature_extractor=None, new_H=2048):
    all_probabilities = []
    all_predictions = []
    all_confidences = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch['image']
            labels = batch['label']
            paths  = batch['path']   # << file names
            inputs = inputs.to(device)

            # inputs = feature_extract(input=inputs, feature_extractor=feature_extractor)
            # inputs = apply_reshape(input=inputs)

            # B, C, H, W = inputs.shape
            # new_H = new_H
            # new_W = (H*W)//new_H
            # inputs = inputs.reshape(B,C,new_H,new_W)

            logits = pipeline.forward_pipeline(inputs)
            probability = F.softmax(input=logits, dim=1)
            predicted_class = torch.argmax(input=probability, dim=1)
            confidence = torch.max(input=probability, dim=1)[0]

            all_predictions.extend(predicted_class.cpu().numpy())
            all_probabilities.extend(probability.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
            all_paths.extend(paths)

            if labels is not None:
                all_labels.extend(labels.cpu().numpy())

    results = {
        'paths': np.array(all_paths),
        'predictions': np.array(all_predictions),
        'probabilities': np.array(all_probabilities),
        'confidences': np.array(all_confidences)
    }

    if all_labels:
        results['labels'] = np.array(all_labels)
    return results

def plot_detailed_metrics(metrics, save_path="detailed_metrics.png"):
    """Create detailed metrics visualization"""
    if metrics is None:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    class_names = metrics['class_names']
    
    # 1. Per-class Precision
    axes[0, 0].bar(class_names, metrics['per_class_precision'])
    axes[0, 0].set_title('Precision per Class')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylim(0, 1)
    
    # 2. Per-class Recall
    axes[0, 1].bar(class_names, metrics['per_class_recall'])
    axes[0, 1].set_title('Recall per Class')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylim(0, 1)
    
    # 3. Per-class F1-Score
    axes[0, 2].bar(class_names, metrics['per_class_f1'])
    axes[0, 2].set_title('F1-Score per Class')
    axes[0, 2].set_ylabel('F1-Score')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].set_ylim(0, 1)
    
    # 4. Support (sample count per class)
    axes[1, 0].bar(class_names, metrics['per_class_support'])
    axes[1, 0].set_title('Support (Sample Count)')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Confusion Matrix
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1, 1])
    axes[1, 1].set_title('Confusion Matrix')
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')
    
    # 6. Metrics comparison
    metric_names = ['Precision', 'Recall', 'F1-Score']
    macro_values = [metrics['precision_macro'], metrics['recall_macro'], metrics['f1_macro']]
    weighted_values = [metrics['precision_weighted'], metrics['recall_weighted'], metrics['f1_weighted']]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    axes[1, 2].bar(x - width/2, macro_values, width, label='Macro Average')
    axes[1, 2].bar(x + width/2, weighted_values, width, label='Weighted Average')
    axes[1, 2].set_title('Macro vs Weighted Averages')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(metric_names)
    axes[1, 2].legend()
    axes[1, 2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Detailed metrics plot saved to: {save_path}")

def evaluate_model(results, class_names=None):
    """Evaluate model performance with comprehensive metrics"""
    if 'labels' not in results:
        logger.warning("No ground truth labels available for evaluation")
        return None
    
    predictions = results['predictions']
    labels = results['labels']
    probabilities = results['probabilities']
    confidences = results['confidences']
    
    # Import additional metrics
    from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score, 
                                roc_curve, precision_recall_curve, average_precision_score)
    
    # Calculate basic metrics
    accuracy = accuracy_score(labels, predictions)
    
    # Calculate precision, recall, f1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    # Calculate macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    # ROC AUC Score
    num_classes = len(np.unique(labels))
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    # Calculate AUC for multiclass
    if num_classes == 2:
        # Binary classification
        auc_score = roc_auc_score(labels, probabilities[:, 1])
        auc_macro = auc_score
        auc_weighted = auc_score
    else:
        # Multiclass classification
        try:
            auc_macro = roc_auc_score(labels, probabilities, multi_class='ovr', average='macro')
            auc_weighted = roc_auc_score(labels, probabilities, multi_class='ovr', average='weighted')
        except ValueError as e:
            logger.warning(f"Could not calculate AUC: {e}")
            auc_macro = np.nan
            auc_weighted = np.nan
    
    # Log overall metrics
    logger.info("=== Overall Performance Metrics ===")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Average Confidence: {np.mean(confidences):.4f}")
    logger.info(f"Macro Precision: {precision_macro:.4f}")
    logger.info(f"Macro Recall: {recall_macro:.4f}")
    logger.info(f"Macro F1-Score: {f1_macro:.4f}")
    logger.info(f"Weighted Precision: {precision_weighted:.4f}")
    logger.info(f"Weighted Recall: {recall_weighted:.4f}")
    logger.info(f"Weighted F1-Score: {f1_weighted:.4f}")
    
    if not np.isnan(auc_macro):
        logger.info(f"Macro AUC-ROC: {auc_macro:.4f}")
        logger.info(f"Weighted AUC-ROC: {auc_weighted:.4f}")
    
    # Log per-class metrics
    logger.info("\n=== Per-Class Metrics ===")
    logger.info(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    logger.info("-" * 60)
    for i, class_name in enumerate(class_names):
        logger.info(f"{class_name:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}")
    
    # Classification report
    report = classification_report(labels, predictions, target_names=class_names)
    logger.info("\nDetailed Classification Report:")
    logger.info(f"\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Create subplots for confusion matrix and ROC curves
    fig = plt.figure(figsize=(16, 6))
    
    # Plot confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Plot ROC curves
    plt.subplot(1, 2, 2)
    if num_classes == 2:
        # Binary ROC curve
        fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    else:
        # Multiclass ROC curves
        from sklearn.preprocessing import label_binarize
        from itertools import cycle
        
        # Binarize labels
        labels_bin = label_binarize(labels, classes=range(num_classes))
        
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])
        for i, color in zip(range(num_classes), colors):
            if np.sum(labels_bin[:, i]) > 0:  # Only plot if class exists
                fpr, tpr, _ = roc_curve(labels_bin[:, i], probabilities[:, i])
                roc_auc = auc_score if num_classes == 2 else roc_auc_score(labels_bin[:, i], probabilities[:, i])
                plt.plot(fpr, tpr, color=color, 
                        label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig('evaluation_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create metrics summary
    metrics_dict = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'auc_macro': auc_macro if not np.isnan(auc_macro) else None,
        'auc_weighted': auc_weighted if not np.isnan(auc_weighted) else None,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'per_class_support': support,
        'confusion_matrix': cm,
        'classification_report': report,
        'avg_confidence': np.mean(confidences),
        'class_names': class_names
    }
    
    return metrics_dict
def create_metrics_summary(metrics, save_path="metrics_summary.txt"):
    """Create and save a detailed metrics summary"""
    if metrics is None:
        return
    
    summary = []
    summary.append("=" * 60)
    summary.append("MODEL EVALUATION SUMMARY")
    summary.append("=" * 60)
    summary.append("")
    
    # Overall metrics
    summary.append("OVERALL PERFORMANCE:")
    summary.append(f"  Accuracy:           {metrics['accuracy']:.4f}")
    summary.append(f"  Average Confidence: {metrics['avg_confidence']:.4f}")
    summary.append("")
    
    # Macro averages
    summary.append("MACRO AVERAGES:")
    summary.append(f"  Precision: {metrics['precision_macro']:.4f}")
    summary.append(f"  Recall:    {metrics['recall_macro']:.4f}")
    summary.append(f"  F1-Score:  {metrics['f1_macro']:.4f}")
    if metrics['auc_macro'] is not None:
        summary.append(f"  AUC-ROC:   {metrics['auc_macro']:.4f}")
    summary.append("")
    
    # Weighted averages
    summary.append("WEIGHTED AVERAGES:")
    summary.append(f"  Precision: {metrics['precision_weighted']:.4f}")
    summary.append(f"  Recall:    {metrics['recall_weighted']:.4f}")
    summary.append(f"  F1-Score:  {metrics['f1_weighted']:.4f}")
    if metrics['auc_weighted'] is not None:
        summary.append(f"  AUC-ROC:   {metrics['auc_weighted']:.4f}")
    summary.append("")
    
    # Per-class breakdown
    summary.append("PER-CLASS BREAKDOWN:")
    summary.append(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    summary.append("-" * 60)
    for i, class_name in enumerate(metrics['class_names']):
        summary.append(f"{class_name:<15} {metrics['per_class_precision'][i]:<10.4f} "
                      f"{metrics['per_class_recall'][i]:<10.4f} {metrics['per_class_f1'][i]:<10.4f} "
                      f"{metrics['per_class_support'][i]:<10}")
    summary.append("")
    
    # Save summary
    summary_text = "\n".join(summary)
    with open(save_path, 'w') as f:
        f.write(summary_text)
    
    logger.info(f"Metrics summary saved to: {save_path}")
    return summary_text


if __name__ == '__main__':
    device = 'cuda'
    class_names = ['fake', 'real']
    ckpt_path = f'checkpoint/Deepfake_Detection_no_encoder_stage2/checkpoint/best.pt'
    checkpoint = torch.load(ckpt_path, map_location=device)
    logger = logging.getLogger(__name__)
    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    json_file = 'data/image_labels_test2.json'
    feature_extractor = timm.create_model('resnet50_clip.openai', pretrained=True).to('cuda')
    #baseline_dataset = BaselineDataset(json_file=json_file, model = model, transformation=transformation)
    baseline_dataset = BaselineDataset(json_file=json_file, transformation=transformation)
    
    pipeline = load_pipeline(checkpoint_path=ckpt_path)
    dataloader = torch.utils.data.DataLoader(
        baseline_dataset,
        batch_size=200,
        num_workers=16,
        shuffle=False
        )
    results = predicted_batch(pipeline, dataloader, device='cuda',feature_extractor=None, new_H=512)
    if 'labels' in results:
        metrics = evaluate_model(results, class_names)
        
        # Print key metrics
        print("=" * 50)
        print("KEY METRICS SUMMARY:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['f1_macro']:.4f}")
        print(f"Weighted F1: {metrics['f1_weighted']:.4f}")
        if metrics['auc_macro'] is not None:
            print(f"Macro AUC-ROC: {metrics['auc_macro']:.4f}")
        print("=" * 50)
        
        # Create detailed visualizations
        plot_detailed_metrics(metrics, "my_model_metrics.png")
        
        # Save summary to file
        summary = create_metrics_summary(metrics, "my_model_summary.txt")

        # Split confidences by true label
        labels = results['labels']
        confidences = results['confidences']
        predictions = results['predictions']
        paths = results['paths']  # make sure you stored paths during eval

        real_confidences = confidences[labels == 1]  # label 1 = real
        fake_confidences = confidences[labels == 0]  # label 0 = fake

        # Print some stats
        print("=" * 50)
        print("Confidence Analysis:")
        print(f"Avg confidence (Real): {real_confidences.mean():.4f}")
        print(f"Avg confidence (Fake): {fake_confidences.mean():.4f}")
        print(f"Min/Max (Real): {real_confidences.min():.4f} / {real_confidences.max():.4f}")
        print(f"Min/Max (Fake): {fake_confidences.min():.4f} / {fake_confidences.max():.4f}")
        print("=" * 50)

        # Per-image save
        df = pd.DataFrame({
            "path": paths,
            "label": ["real" if l == 1 else "fake" for l in labels],
            "prediction": ["real" if p == 1 else "fake" for p in predictions],
            "confidence": confidences
        })
        df.to_csv("per_image_confidences.csv", index=False)
        print("Saved detailed per-image confidence scores to per_image_confidences.csv")

        # ====================================================
        # Assign generator indices
        # ====================================================
        df = df.copy()

        # Extract generator name for fake, set "real" for real
        df.loc[df["label"] == "fake", "generator"] = df.loc[df["label"] == "fake", "path"].apply(lambda x: x.split("/")[-2])
        df.loc[df["label"] == "real", "generator"] = "real"

        # Get sorted unique fake generators
        generator_names = sorted(df[df["label"] == "fake"]["generator"].unique())

        # Map generators to indices
        gen_to_idx = {gen: idx for idx, gen in enumerate(generator_names)}
        real_index = len(generator_names)  # last index reserved for real
        gen_to_idx["real"] = real_index

        # Assign index
        df["index"] = df["generator"].map(gen_to_idx)

        # ====================================================
        # Compute per-generator scores
        # ====================================================
        real_score_map = {}
        fake_score_map = {}

        # include fake generators + real
        for gen in generator_names + ["real"]:
            subset = df[df["generator"] == gen]
            fake_score = (subset["prediction"] == "fake").mean()
            real_score = (subset["prediction"] == "real").mean()
            real_score_map[gen_to_idx[gen]] = real_score
            fake_score_map[gen_to_idx[gen]] = fake_score

        # ====================================================
        # Create per-image rows (7800 rows)
        # ====================================================
        df["real_score"] = df["index"].map(real_score_map)
        df["fake_score"] = df["index"].map(fake_score_map)

        # only keep required columns
        final_df = df[["index", "real_score", "fake_score"]]

        # Save
        final_df.to_csv("generator_score.csv", index=False)
        np.save("generator_score.npy", final_df.to_numpy())

        print("Saved generator_score.csv and generator_score.npy with 7800 rows")
        
    else:
        print("No ground truth labels available for evaluation")
    
