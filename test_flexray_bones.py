import os
import time
import cv2
import numpy as np
import pandas as pd
import torch
from fxr.inference import FleXraySegmenter


def unpad_and_resize_mask(mask_256, orig_shape):
    """
    Inverts FleXray's square-padding and bilinear resizing contract:
    1. FleXray pads input (H_orig, W_orig) to square (side, side) where side = max(H_orig, W_orig).
       pad_left = (side - W_orig) // 2, pad_top = (side - H_orig) // 2
    2. Resizes square to (256, 256).
    
    Inverse:
    1. Upsample 256x256 mask back to (side, side) using nearest-neighbor interpolation.
    2. Crop the unpadded central region [pad_top : pad_top + H_orig, pad_left : pad_left + W_orig].
    Result matches orig_shape (H_orig, W_orig) pixel-for-pixel.
    """
    h_orig, w_orig = orig_shape
    side = max(h_orig, w_orig)
    pad_left = (side - w_orig) // 2
    pad_top = (side - h_orig) // 2

    # Upsample to side x side
    if mask_256.ndim == 2:
        mask_square = cv2.resize(mask_256, (side, side), interpolation=cv2.INTER_NEAREST)
        mask_orig = mask_square[pad_top : pad_top + h_orig, pad_left : pad_left + w_orig]
    else:
        # Batch of channels [C, 256, 256]
        c = mask_256.shape[0]
        mask_orig = np.zeros((c, h_orig, w_orig), dtype=mask_256.dtype)
        for i in range(c):
            ch_square = cv2.resize(mask_256[i], (side, side), interpolation=cv2.INTER_NEAREST)
            mask_orig[i] = ch_square[pad_top : pad_top + h_orig, pad_left : pad_left + w_orig]

    return mask_orig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading FleXray Segmenter on {device}...")
    segmenter = FleXraySegmenter.from_pretrained(device=device)

    all_labels = segmenter.label_names
    non_bone = {'background', 'kidneys', 'liver', 'spleen', 'lungs', 'heart'}
    bone_indices = [i for i, l in enumerate(all_labels) if l not in non_bone]
    print(f"Total FleXray labels: {len(all_labels)} | Bone labels: {len(bone_indices)}")

    # Load 20 images from test split
    test_csv = "Utilities/Fracture Split/test.csv"
    df_test = pd.read_csv(test_csv)
    image_list = df_test["image_id"].tolist()[:20]
    print(f"Running coordinate-aligned FleXray inference on {len(image_list)} FracAtlas test images...\n")

    out_dir = "results/flexray_bone_segmentations"
    os.makedirs(out_dir, exist_ok=True)

    summary_records = []

    # Consistent distinct colors for bone anatomies
    np.random.seed(42)
    colors = np.random.randint(60, 255, size=(len(all_labels), 3))

    for idx, img_name in enumerate(image_list):
        img_path = os.path.join("images/Fractured", img_name)
        if not os.path.exists(img_path):
            continue

        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join("fracture_masks", f"{base_name}.png")

        raw_bgr = cv2.imread(img_path)
        h_orig, w_orig = raw_bgr.shape[:2]

        t0 = time.time()
        # FleXray prediction on square canvas
        pred = segmenter.predict(img_path, threshold=0.5)
        inference_time = time.time() - t0

        masks_256 = pred.masks[0].cpu().numpy() # [61, 256, 256]

        # Invert the letterbox/pillarbox square padding directly to native resolution (H_orig, W_orig)
        masks_aligned = unpad_and_resize_mask(masks_256, (h_orig, w_orig))

        # Aggregate unified bone mask at native resolution
        bone_channels = masks_aligned[bone_indices, :, :]
        unified_bone = (np.sum(bone_channels, axis=0) > 0).astype(np.uint8)

        # Detect active bone anatomies
        active_bones = []
        for bi in bone_indices:
            if masks_aligned[bi].sum() > 50:
                active_bones.append(all_labels[bi])

        # Load ground truth fracture mask at native resolution
        gt_fracture = np.zeros((h_orig, w_orig), dtype=np.uint8)
        if os.path.exists(mask_path):
            gt_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_raw.shape != (h_orig, w_orig):
                gt_raw = cv2.resize(gt_raw, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            gt_fracture = (gt_raw > 0).astype(np.uint8)

        # Calculate exact pixel metrics at native coordinates
        total_pixels = h_orig * w_orig
        bone_pixels = int(unified_bone.sum())
        bone_ratio = (bone_pixels / total_pixels) * 100.0

        fracture_pixels = int(gt_fracture.sum())
        if fracture_pixels > 0:
            fracture_in_bone = int(np.logical_and(gt_fracture == 1, unified_bone == 1).sum())
            containment = (fracture_in_bone / fracture_pixels) * 100.0
        else:
            containment = 100.0

        summary_records.append({
            "image_id": img_name,
            "orig_width": w_orig,
            "orig_height": h_orig,
            "bone_coverage_pct": round(bone_ratio, 2),
            "fracture_pixels": fracture_pixels,
            "fracture_in_bone_pct": round(containment, 2),
            "detected_bones": ", ".join(active_bones[:4]),
            "inference_time_sec": round(inference_time, 3)
        })

        # Generate 4-Panel Diagnostic Visualization at clean standardized scale (e.g. 512 height)
        disp_h = 512
        disp_w = int(w_orig * (disp_h / h_orig))

        # 1. Native Raw X-ray
        raw_disp = cv2.resize(raw_bgr, (disp_w, disp_h))

        # 2. Multi-Anatomy Overlay (Perfect Alignment)
        anatomy_disp = raw_disp.copy()
        for bi in bone_indices:
            b_mask = masks_aligned[bi]
            if b_mask.sum() > 20:
                b_mask_disp = cv2.resize(b_mask, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
                color = colors[bi].tolist()
                anatomy_disp[b_mask_disp > 0] = (anatomy_disp[b_mask_disp > 0] * 0.4 + np.array(color) * 0.6).astype(np.uint8)

        # 3. Unified Bone Mask Overlay
        bone_disp_mask = cv2.resize(unified_bone, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
        bone_vis = raw_disp.copy()
        # Soft blue/cyan bone mask overlay
        bone_vis[bone_disp_mask == 1] = (bone_vis[bone_disp_mask == 1] * 0.45 + np.array([220, 180, 50]) * 0.55).astype(np.uint8)

        # 4. Multi-Class Diagnostic Composite
        # Background: untouched radiograph
        # Bone: soft cyan
        # Fracture: bright red / yellow on overlap
        gt_disp_mask = cv2.resize(gt_fracture, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
        multi_vis = raw_disp.copy()
        multi_vis[bone_disp_mask == 1] = (multi_vis[bone_disp_mask == 1] * 0.5 + np.array([200, 170, 40]) * 0.5).astype(np.uint8)
        multi_vis[gt_disp_mask == 1] = [0, 0, 255] # Red fracture

        # Horizontal stack of panels
        row = np.hstack([raw_disp, anatomy_disp, bone_vis, multi_vis])

        # Header bar
        header_bar = np.zeros((35, row.shape[1], 3), dtype=np.uint8)
        cv2.putText(header_bar, f"1. Raw X-ray ({w_orig}x{h_orig})", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        cv2.putText(header_bar, "2. FleXray Multi-Anatomy", (disp_w + 10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        cv2.putText(header_bar, "3. Unified Bone vs Background", (2*disp_w + 10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        cv2.putText(header_bar, "4. Multi-Class (Bone+Fracture)", (3*disp_w + 10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

        final_img = np.vstack([header_bar, row])
        save_path = os.path.join(out_dir, f"flexray_{base_name}.png")
        cv2.imwrite(save_path, final_img)

        print(f"[{idx+1:02d}/20] {img_name} ({w_orig}x{h_orig}) -> Bone: {bone_ratio:.1f}% | Fracture in Bone: {containment:.1f}% | Bones: {', '.join(active_bones[:3])}")

    df_summary = pd.DataFrame(summary_records)
    summary_path = "results/flexray_test_20_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print("\n" + "=" * 85)
    print(f"COORDINATE-ALIGNED FLEXRAY TESTING COMPLETE ON 20 FRACATLAS IMAGES!")
    print(f"Mean Bone Coverage: {df_summary['bone_coverage_pct'].mean():.2f}%")
    print(f"Mean Fracture Containment: {df_summary['fracture_in_bone_pct'].mean():.2f}%")
    print(f"Summary saved to: {summary_path}")
    print(f"Visual overlays saved to: {out_dir}")
    print("=" * 85)
    print(df_summary[["image_id", "orig_width", "orig_height", "bone_coverage_pct", "fracture_in_bone_pct", "detected_bones"]].to_string(index=False))


if __name__ == "__main__":
    main()
