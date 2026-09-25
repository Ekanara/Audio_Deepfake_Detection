
from datasets import load_dataset
from collections import defaultdict
from PIL import Image, ImageDraw
import os
import cv2
import random
import numpy as np

DATASET_NAME = "ArtMancer/ORiDa-train-only-with-SDcaptions-1"
OUTPUT_DIR = "organized_dataset"
SETS_PER_GROUP = 3
TRAIN_RATIO = 0.9
LOW_THRESH = 100
HIGH_THRESH = 200

def create_folders():
    for split in ["train", "val"]:
        for f in ["background", "target", "reference_image", "canny_edge", "mask_on_background"]:
            os.makedirs(f"{OUTPUT_DIR}/{split}/{f}", exist_ok=True)

def apply_mask_to_background(background_img, mask_img):
    """Apply a black mask onto the background image."""
    # Convert images to numpy arrays
    bg_array = np.array(background_img.convert("RGB"))
    mask_array = np.array(mask_img.convert("L"))  # Convert mask to grayscale
    
    # Create a copy of the background
    result = bg_array.copy()
    
    # Where mask is white (> 127), make the background black
    mask_binary = mask_array > 127
    result[mask_binary] = [0, 0, 0]  # Black
    
    return Image.fromarray(result)

def save_set(background, targets, target_start, split):
    bg_img = background["image"]
    saved = 0
    
    for i, t in enumerate(targets):
        tid = target_start + i
        bg_path = f"{OUTPUT_DIR}/{split}/background/{tid}.png"
        tgt_path = f"{OUTPUT_DIR}/{split}/target/{tid}.png"
        ref_path = f"{OUTPUT_DIR}/{split}/reference_image/{tid}.png"
        cap_path = f"{OUTPUT_DIR}/{split}/target/{tid}.txt"
        canny_path = f"{OUTPUT_DIR}/{split}/canny_edge/{tid}.png"
        mask_bg_path = f"{OUTPUT_DIR}/{split}/mask_on_background/{tid}.png"
        
        # background (duplicated)
        bg_img.save(bg_path)
        
        # target
        t["image"].save(tgt_path)
        
        # reference
        t["ground_truth"].save(ref_path)
        
        # caption
        caption = t.get("caption") or ""
        with open(cap_path, "w", encoding="utf-8") as f:
            f.write(caption)
        
        # canny edge (from target)
        gray = cv2.imread(tgt_path, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(gray, LOW_THRESH, HIGH_THRESH)
        cv2.imwrite(canny_path, edges)
        
        # mask on background - apply the actual mask from the dataset
        mask_img = t.get("masks")
        if mask_img is not None:
            masked_bg = apply_mask_to_background(bg_img, mask_img)
            masked_bg.save(mask_bg_path)
        else:
            # If no mask available, save a copy of the background
            bg_img.save(mask_bg_path)
        
        saved += 1
    
    return saved

def main():
    create_folders()
    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True
    )
    
    current_bg = None
    current_targets = []
    group_set_count = defaultdict(int)
    target_id = 1
    
    for row in dataset:
        if row["type"] != 1:
            continue
        
        group_id = row["group_id"]
        bbox = row.get("bbox", [])
        has_mask = bbox and len(bbox) > 0
        
        # new background → flush previous set
        if not has_mask:
            if (
                current_bg is not None
                and current_targets
                and group_set_count[group_id] < SETS_PER_GROUP
            ):
                split = "train" if random.random() < TRAIN_RATIO else "val"
                saved = save_set(current_bg, current_targets, target_id, split)
                target_id += saved
                group_set_count[group_id] += 1
            
            current_bg = row
            current_targets = []
        else:
            if current_bg is not None:
                current_targets.append(row)
    
    # flush last set
    if current_bg and current_targets:
        split = "train" if random.random() < TRAIN_RATIO else "val"
        save_set(current_bg, current_targets, target_id, split)

if __name__ == "__main__":
    main()