"""
Fast & Coordinate-Aligned Batch Generation of Bone Masks and 3-Class Semantic Masks
for all 719 FracAtlas Fractured Scans using FleXray Foundation Model.

Optimization:
Aggregates the 55 bone channels at 256x256 first, then performs a single coordinate-aligned
unpadding and nearest-neighbor upsample back to native resolution (H_orig, W_orig).
Mathematically identical, but ~50x faster (only 1 resize instead of 61 per image).

Output Formats:
- bone_masks/<image_name>.png: Binary mask (0: Non-bone, 255: Bone) at native resolution.
- multiclass_masks/<image_name>.png: 3-class uint8 mask at native resolution:
    - 0: Background / Soft Tissue
    - 1: Intact Bone
    - 2: Fracture
"""

import os
import time
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from fxr.inference import FleXraySegmenter


def unpad_and_resize_single_mask(mask_256, orig_shape):
    """
    Exact inverse of FleXray's square padding and downsampling for a single 2D mask:
    1. Upsamples 256x256 mask back to (side, side) where side = max(H_orig, W_orig) using nearest neighbor.
    2. Crops the central region [pad_top : pad_top + H_orig, pad_left : pad_left + W_orig].
    """
    h_orig, w_orig = orig_shape
    side = max(h_orig, w_orig)
    pad_left = (side - w_orig) // 2
    pad_top = (side - h_orig) // 2

    mask_square = cv2.resize(mask_256, (side, side), interpolation=cv2.INTER_NEAREST)
    return mask_square[pad_top : pad_top + h_orig, pad_left : pad_left + w_orig]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading FleXray Segmenter on {device}...")
    segmenter = FleXraySegmenter.from_pretrained(device=device)

    all_labels = segmenter.label_names
    non_bone = {'background', 'kidneys', 'liver', 'spleen', 'lungs', 'heart'}
    bone_indices = [i for i, l in enumerate(all_labels) if l not in non_bone]
    print(f"Total FleXray labels: {len(all_labels)} | Bone labels: {len(bone_indices)}")

    # Directories
    img_dir = "images/Fractured"
    frac_mask_dir = "fracture_masks"
    bone_mask_dir = "bone_masks"
    multi_mask_dir = "multiclass_masks"

    os.makedirs(bone_mask_dir, exist_ok=True)
    os.makedirs(multi_mask_dir, exist_ok=True)

    # Collect all fractured images across train, valid, test
    splits = ["train", "valid", "test"]
    all_records = []
    for sp in splits:
        csv_p = f"Utilities/Fracture Split/{sp}.csv"
        if os.path.exists(csv_p):
            df_sp = pd.read_csv(csv_p)
            for _, r in df_sp.iterrows():
                all_records.append({"image_id": r["image_id"], "split": sp})

    df_all = pd.DataFrame(all_records).drop_duplicates(subset=["image_id"]).reset_index(drop=True)
    total_imgs = len(df_all)
    print(f"Found {total_imgs} images across official splits (train={len(df_all[df_all['split']=='train'])}, valid={len(df_all[df_all['split']=='valid'])}, test={len(df_all[df_all['split']=='test'])}).")

    stats = []
    t_start = time.time()

    for idx, row in tqdm(df_all.iterrows(), total=total_imgs, desc="Generating Bone & 3-Class Masks"):
        img_name = row["image_id"]
        split = row["split"]
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            continue

        base_name = os.path.splitext(img_name)[0]
        frac_mask_path = os.path.join(frac_mask_dir, f"{base_name}.png")

        raw_bgr = cv2.imread(img_path)
        if raw_bgr is None:
            continue
        h_orig, w_orig = raw_bgr.shape[:2]

        # FleXray prediction on 256x256 square canvas
        pred = segmenter.predict(img_path, threshold=0.5)
        masks_256 = pred.masks[0].cpu().numpy() # [61, 256, 256]

        # Aggregate bone channels at 256x256 first
        bone_channels_256 = masks_256[bone_indices, :, :]
        bone_bin_256 = (np.sum(bone_channels_256, axis=0) > 0).astype(np.uint8)

        # Unpad and resize ONLY the single aggregated bone mask to native resolution
        bone_bin = unpad_and_resize_single_mask(bone_bin_256, (h_orig, w_orig))

        # Save binary bone mask (255 for bone, 0 for non-bone)
        bone_save_p = os.path.join(bone_mask_dir, f"{base_name}.png")
        cv2.imwrite(bone_save_p, bone_bin * 255)

        # Load Fracture ground truth if present
        frac_bin = np.zeros((h_orig, w_orig), dtype=np.uint8)
        if os.path.exists(frac_mask_path):
            fm = cv2.imread(frac_mask_path, cv2.IMREAD_GRAYSCALE)
            if fm is not None:
                frac_bin = (fm > 127).astype(np.uint8)

        # Create 3-Class Mask:
        # 0: Background
        # 1: Intact Bone (bone_bin == 1 and frac_bin == 0)
        # 2: Fracture (frac_bin == 1)
        multi_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        multi_mask[bone_bin == 1] = 1
        multi_mask[frac_bin == 1] = 2  # Fracture takes priority

        multi_save_p = os.path.join(multi_mask_dir, f"{base_name}.png")
        cv2.imwrite(multi_save_p, multi_mask)

        # Compute overlap & stats
        bone_pix = int(bone_bin.sum())
        frac_pix = int(frac_bin.sum())
        frac_in_bone = int((frac_bin & bone_bin).sum())
        containment = (frac_in_bone / frac_pix * 100.0) if frac_pix > 0 else 100.0
        bone_cov = (bone_pix / (h_orig * w_orig)) * 100.0

        stats.append({
            "image_id": img_name,
            "split": split,
            "width": w_orig,
            "height": h_orig,
            "bone_pixels": bone_pix,
            "fracture_pixels": frac_pix,
            "fracture_in_bone_pixels": frac_in_bone,
            "fracture_in_bone_pct": round(containment, 2),
            "bone_coverage_pct": round(bone_cov, 2)
        })

    elapsed = time.time() - t_start
    print(f"\nCompleted in {elapsed:.2f} seconds ({elapsed/total_imgs:.3f} s/image, {total_imgs/elapsed:.1f} FPS)!")

    df_stats = pd.DataFrame(stats)
    summary_path = "results/dataset_3class_summary.csv"
    df_stats.to_csv(summary_path, index=False)
    print(f"Summary metrics saved to {summary_path}")
    print(f"Mean Bone Coverage: {df_stats['bone_coverage_pct'].mean():.2f}%")
    print(f"Mean Fracture-in-Bone Containment: {df_stats['fracture_in_bone_pct'].mean():.2f}%")


if __name__ == "__main__":
    main()
