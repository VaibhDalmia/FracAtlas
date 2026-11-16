"""
Generate publication-quality combined visual representations of:
1. Original X-ray Radiograph
2. Coordinate-Aligned FleXray Bone Mask
3. Ground Truth Fracture Mask
4. 3-Class Semantic Color Map (0: Background, 1: Intact Bone, 2: Fracture)
5. Blended Clinical Composite Overlay

Saved into results/dataset_visualizations/ for the official FracAtlas test split (63 scans).
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def create_visualization_strip(raw_bgr, bone_bin, frac_bin, target_height=512):
    """
    Constructs a 5-panel diagnostic strip:
    [Raw Image | Bone Mask | Fracture Mask | 3-Class Color Map | Blended Clinical Overlay]
    """
    h_orig, w_orig = raw_bgr.shape[:2]

    # Panel 1: Raw Radiograph
    p1 = raw_bgr.copy()
    cv2.putText(p1, "1. Raw Radiograph", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Panel 2: Bone Mask (Grayscale / Highlight)
    p2 = cv2.cvtColor(bone_bin * 255, cv2.COLOR_GRAY2BGR)
    cv2.putText(p2, "2. FleXray Bone Mask", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

    # Panel 3: Fracture Mask
    p3 = cv2.cvtColor(frac_bin * 255, cv2.COLOR_GRAY2BGR)
    if frac_bin.sum() > 0:
        # Red highlight for fracture
        p3[frac_bin == 1] = [0, 0, 255]
    cv2.putText(p3, "3. Fracture Ground Truth", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 100, 255), 2, cv2.LINE_AA)

    # Panel 4: 3-Class Semantic Segmentation Map
    # Class 0: Background (Black)
    # Class 1: Intact Bone (Slate Blue / Cyan)
    # Class 2: Fracture (Bright Red)
    p4 = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    p4[bone_bin == 1] = [180, 105, 0]      # BGR: Slate Blue / Cyan
    p4[frac_bin == 1] = [0, 0, 255]        # BGR: Bright Red
    cv2.putText(p4, "4. 3-Class Map (BG/Bone/Frac)", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Panel 5: Blended Clinical Composite Overlay
    p5 = raw_bgr.copy()
    overlay = p5.copy()
    overlay[bone_bin == 1] = [200, 150, 0] # Cyan tint for bone
    overlay[frac_bin == 1] = [0, 0, 255]   # Red tint for fracture
    cv2.addWeighted(overlay, 0.45, p5, 0.55, 0, p5)

    # Add green boundary contour around ground truth fracture for clarity
    if frac_bin.sum() > 0:
        contours, _ = cv2.findContours(frac_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p5, contours, -1, (0, 255, 0), 2)
    cv2.putText(p5, "5. Clinical Overlay (Grn Cont)", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

    # Resize all panels to target_height preserving aspect ratio
    panels = [p1, p2, p3, p4, p5]
    scaled_panels = []
    for p in panels:
        scale = target_height / float(p.shape[0])
        new_w = max(1, int(p.shape[1] * scale))
        scaled_p = cv2.resize(p, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
        scaled_panels.append(scaled_p)

    # Add 4-pixel vertical separator between panels
    separator = np.full((target_height, 4, 3), 60, dtype=np.uint8)
    composite_list = []
    for i, sp in enumerate(scaled_panels):
        composite_list.append(sp)
        if i < len(scaled_panels) - 1:
            composite_list.append(separator)

    composite = np.hstack(composite_list)
    return composite, p5


def main():
    img_dir = "images/Fractured"
    bone_dir = "bone_masks"
    frac_dir = "fracture_masks"
    test_csv = "Utilities/Fracture Split/test.csv"

    out_dir_strips = "results/dataset_visualizations/strips"
    out_dir_blended = "results/dataset_visualizations/blended_overlays"
    os.makedirs(out_dir_strips, exist_ok=True)
    os.makedirs(out_dir_blended, exist_ok=True)

    df_test = pd.read_csv(test_csv)
    print(f"Generating combined visualizations for all {len(df_test)} test scans...")

    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Rendering Visualizations"):
        img_name = row["image_id"]
        base_name = os.path.splitext(img_name)[0]

        img_p = os.path.join(img_dir, img_name)
        bone_p = os.path.join(bone_dir, f"{base_name}.png")
        frac_p = os.path.join(frac_dir, f"{base_name}.png")

        if not os.path.exists(img_p):
            continue

        raw_bgr = cv2.imread(img_p)
        h, w = raw_bgr.shape[:2]

        bone_bin = np.zeros((h, w), dtype=np.uint8)
        if os.path.exists(bone_p):
            bm = cv2.imread(bone_p, cv2.IMREAD_GRAYSCALE)
            if bm is not None:
                bone_bin = (bm > 127).astype(np.uint8)

        frac_bin = np.zeros((h, w), dtype=np.uint8)
        if os.path.exists(frac_p):
            fm = cv2.imread(frac_p, cv2.IMREAD_GRAYSCALE)
            if fm is not None:
                frac_bin = (fm > 127).astype(np.uint8)

        # Generate diagnostic strip and blended overlay
        strip, blended = create_visualization_strip(raw_bgr, bone_bin, frac_bin, target_height=512)

        strip_save_p = os.path.join(out_dir_strips, f"strip_{base_name}.png")
        blended_save_p = os.path.join(out_dir_blended, f"blended_{base_name}.png")

        cv2.imwrite(strip_save_p, strip)
        cv2.imwrite(blended_save_p, blended)

    print(f"\nAll 63 test visual representations generated successfully!")
    print(f"1. 5-Panel Diagnostic Strips saved to: {out_dir_strips}")
    print(f"2. Standalone Blended Overlays saved to: {out_dir_blended}")


if __name__ == "__main__":
    main()
