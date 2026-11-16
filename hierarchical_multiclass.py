"""
Hierarchical Multi-Class Segmentation Pipeline:
Combines FleXray (MIT CSAIL) for Bone Segmentation with Boundary-Aware Attention U-Net for Fracture Segmentation.

Produces 3-Class Semantic Segmentation Masks:
- Class 0: Background / Soft Tissue
- Class 1: Intact Bone
- Class 2: Bone Fracture

Applies Anatomical Prior Filtering: Fractures can only occur within skeletal bones (M_fracture = M_fracture & M_bone),
eliminating false positives on soft tissues, foreign metal markers, and image borders.
"""

import os
import time
import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fxr.inference import FleXraySegmenter

from attention_unet import AttentionUNet


def unpad_and_resize_single_mask(mask_256, orig_shape):
    """
    Exact inverse of FleXray's square padding and downsampling for a 2D mask:
    1. Upsamples 256x256 mask back to (side, side) where side = max(H_orig, W_orig) using nearest-neighbor.
    2. Crops the central region [pad_top : pad_top + H_orig, pad_left : pad_left + W_orig].
    """
    h_orig, w_orig = orig_shape
    side = max(h_orig, w_orig)
    pad_left = (side - w_orig) // 2
    pad_top = (side - h_orig) // 2

    mask_square = cv2.resize(mask_256, (side, side), interpolation=cv2.INTER_NEAREST)
    return mask_square[pad_top : pad_top + h_orig, pad_left : pad_left + w_orig]


def preprocess_for_unet(img_gray, target_size=(512, 512)):
    """Applies CLAHE and normalizes for Attention U-Net."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    resized = cv2.resize(enhanced, target_size, interpolation=cv2.INTER_LINEAR)
    norm = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0) # [1, 1, 512, 512]
    return tensor


def compute_hd95_single(pred_mask, gt_mask):
    """Computes 95th percentile Hausdorff Distance (HD95) between binary masks in pixels."""
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)

    if np.sum(pred) == 0 and np.sum(gt) == 0:
        return 0.0
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        h, w = pred.shape
        return float(np.sqrt(h**2 + w**2))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    boundary_pred = cv2.morphologyEx(pred, cv2.MORPH_GRADIENT, kernel)
    boundary_gt = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, kernel)

    dist_to_gt = cv2.distanceTransform(1 - boundary_gt, cv2.DIST_L2, 3)
    pred_pts_dist = dist_to_gt[boundary_pred > 0]

    dist_to_pred = cv2.distanceTransform(1 - boundary_pred, cv2.DIST_L2, 3)
    gt_pts_dist = dist_to_pred[boundary_gt > 0]

    if len(pred_pts_dist) == 0 or len(gt_pts_dist) == 0:
        return 0.0

    hd95_p2g = np.percentile(pred_pts_dist, 95)
    hd95_g2p = np.percentile(gt_pts_dist, 95)
    return float(max(hd95_p2g, hd95_g2p))


class HierarchicalMultiClassSegmenter:
    def __init__(self, unet_checkpoint="checkpoints/best_boundary_unet_512.pth", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Hierarchical Pipeline] Initializing on device: {self.device}")

        # 1. Load FleXray Segmenter
        print("[Hierarchical Pipeline] Loading FleXray Foundation Model...")
        self.flexray = FleXraySegmenter.from_pretrained(device=self.device)
        non_bone = {'background', 'kidneys', 'liver', 'spleen', 'lungs', 'heart'}
        self.bone_indices = [i for i, l in enumerate(self.flexray.label_names) if l not in non_bone]
        print(f"[Hierarchical Pipeline] FleXray loaded: {len(self.bone_indices)} bone classes registered.")

        # 2. Load Boundary-Aware Attention U-Net
        print(f"[Hierarchical Pipeline] Loading Boundary-Aware Attention U-Net from {unet_checkpoint}...")
        self.unet = AttentionUNet(in_channels=1, out_channels=1, base_features=32).to(self.device)
        state_dict = torch.load(unet_checkpoint, map_location=self.device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self.unet.load_state_dict(state_dict)
        self.unet.eval()
        print("[Hierarchical Pipeline] Attention U-Net loaded and ready.")

    @torch.no_grad()
    def segment(self, img_path, fracture_threshold=0.5, apply_anatomical_filter=True):
        t0 = time.time()
        raw_bgr = cv2.imread(img_path)
        if raw_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        h_orig, w_orig = raw_bgr.shape[:2]
        img_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)

        # Stage 1: FleXray Bone Segmentation
        t_flex0 = time.time()
        flex_pred = self.flexray.predict(img_path, threshold=0.5)
        masks_256 = flex_pred.masks[0].cpu().numpy() # [61, 256, 256]
        # Aggregate 55 bone channels first at 256x256, then unpad and resize single binary bone mask
        bone_channels_256 = masks_256[self.bone_indices, :, :]
        bone_bin_256 = (np.sum(bone_channels_256, axis=0) > 0).astype(np.uint8)
        bone_mask = unpad_and_resize_single_mask(bone_bin_256, (h_orig, w_orig))
        t_flex = time.time() - t_flex0

        # Stage 2: Attention U-Net Fracture Segmentation
        t_unet0 = time.time()
        unet_input = preprocess_for_unet(img_gray, target_size=(512, 512)).to(self.device)
        logits = self.unet(unet_input)
        prob_512 = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred_512 = (prob_512 > fracture_threshold).astype(np.uint8)
        # Resize predicted fracture mask to native resolution
        raw_fracture_mask = cv2.resize(pred_512, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        t_unet = time.time() - t_unet0

        # Stage 3: Anatomical Prior Filtering & Hierarchical Fusion
        t_fuse0 = time.time()
        if apply_anatomical_filter:
            filtered_fracture_mask = raw_fracture_mask & bone_mask
        else:
            filtered_fracture_mask = raw_fracture_mask

        # Construct 3-Class semantic segmentation mask:
        # 0: Background, 1: Intact Bone, 2: Fracture
        multiclass_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        multiclass_mask[bone_mask == 1] = 1
        multiclass_mask[filtered_fracture_mask == 1] = 2
        t_fuse = time.time() - t_fuse0

        total_time = time.time() - t0
        timing_info = {
            "total_sec": total_time,
            "flexray_sec": t_flex,
            "unet_sec": t_unet,
            "fusion_sec": t_fuse
        }

        return multiclass_mask, bone_mask, raw_fracture_mask, filtered_fracture_mask, timing_info


def create_diagnostic_composite(raw_bgr, bone_mask, filtered_fracture_mask, multiclass_mask, gt_fracture_mask=None):
    h, w = raw_bgr.shape[:2]

    # Panel 1: Raw
    p1 = raw_bgr.copy()
    cv2.putText(p1, "Raw Radiograph", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Panel 2: Bone Segmentation
    p2 = raw_bgr.copy()
    overlay_bone = p2.copy()
    overlay_bone[bone_mask == 1] = [230, 216, 0] # Cyan in BGR
    cv2.addWeighted(overlay_bone, 0.45, p2, 0.55, 0, p2)
    cv2.putText(p2, "FleXray Bone (Aligned)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Panel 3: Fracture Mask
    p3 = raw_bgr.copy()
    if gt_fracture_mask is not None and gt_fracture_mask.sum() > 0:
        contours, _ = cv2.findContours(gt_fracture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p3, contours, -1, (0, 255, 0), 2)
    overlay_frac = p3.copy()
    overlay_frac[filtered_fracture_mask == 1] = [0, 0, 255] # Red
    cv2.addWeighted(overlay_frac, 0.6, p3, 0.4, 0, p3)
    cv2.putText(p3, "Fracture (Pred: Red, GT: Grn)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Panel 4: 3-Class Color Map
    p4 = np.zeros((h, w, 3), dtype=np.uint8)
    p4[multiclass_mask == 1] = [180, 105, 0] # Slate Blue/Cyan for Bone
    p4[multiclass_mask == 2] = [0, 0, 255]   # Red for Fracture
    cv2.putText(p4, "3-Class Map (0:BG, 1:Bone, 2:Frac)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Panel 5: Clinical Composite
    p5 = raw_bgr.copy()
    comp_overlay = p5.copy()
    comp_overlay[multiclass_mask == 1] = [200, 150, 0]
    comp_overlay[multiclass_mask == 2] = [0, 0, 255]
    cv2.addWeighted(comp_overlay, 0.5, p5, 0.5, 0, p5)
    cv2.putText(p5, "Multi-Class Clinical Overlay", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    target_h = 450
    panels = [p1, p2, p3, p4, p5]
    resized_panels = []
    for p in panels:
        scale = target_h / p.shape[0]
        new_w = int(p.shape[1] * scale)
        resized_panels.append(cv2.resize(p, (new_w, target_h), interpolation=cv2.INTER_LINEAR))

    composite = np.hstack(resized_panels)
    return composite


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Multi-Class Segmentation on FracAtlas")
    parser.add_argument("--num_samples", type=int, default=63, help="Number of test images to evaluate")
    parser.add_argument("--out_dir", type=str, default="results/hierarchical_multiclass", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    segmenter = HierarchicalMultiClassSegmenter()

    test_csv = "Utilities/Fracture Split/test.csv"
    df_test = pd.read_csv(test_csv)
    image_list = df_test["image_id"].tolist()[:args.num_samples]

    print(f"\nProcessing {len(image_list)} test images through Hierarchical Multi-Class Pipeline...\n")
    results = []

    # Accumulate metrics across all test images
    all_pred_frac, all_gt_frac = [], []
    all_pred_bone, all_gt_bone = [], []
    all_pred_bg, all_gt_bg = [], []
    hd95_list = []

    for i, img_name in enumerate(image_list):
        img_path = os.path.join("images/Fractured", img_name)
        if not os.path.exists(img_path):
            continue
        base_name = os.path.splitext(img_name)[0]
        gt_mask_path = os.path.join("fracture_masks", f"{base_name}.png")
        multi_gt_path = os.path.join("multiclass_masks", f"{base_name}.png")

        raw_bgr = cv2.imread(img_path)
        h, w = raw_bgr.shape[:2]

        gt_frac_mask = np.zeros((h, w), dtype=np.uint8)
        if os.path.exists(gt_mask_path):
            fm = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
            if fm is not None:
                gt_frac_mask = (fm > 127).astype(np.uint8)

        gt_multi = np.zeros((h, w), dtype=np.uint8)
        if os.path.exists(multi_gt_path):
            gm = cv2.imread(multi_gt_path, cv2.IMREAD_GRAYSCALE)
            if gm is not None:
                gt_multi = gm

        multiclass_mask, bone_mask, raw_frac, filtered_frac, timing = segmenter.segment(img_path)

        # Bone coverage & containment
        bone_coverage = (bone_mask.sum() / (h * w)) * 100.0
        raw_frac_pixels = raw_frac.sum()
        filtered_frac_pixels = filtered_frac.sum()
        fp_filtered_out = raw_frac_pixels - filtered_frac_pixels

        # Fracture metrics
        tp_f = int((filtered_frac & gt_frac_mask).sum())
        fp_f = int((filtered_frac & (1 - gt_frac_mask)).sum())
        fn_f = int(((1 - filtered_frac) & gt_frac_mask).sum())

        d_frac = (2.0 * tp_f + 1e-5) / (2.0 * tp_f + fp_f + fn_f + 1e-5)
        iou_frac = (tp_f + 1e-5) / (tp_f + fp_f + fn_f + 1e-5)
        prec_frac = (tp_f + 1e-5) / (tp_f + fp_f + 1e-5)
        rec_frac = (tp_f + 1e-5) / (tp_f + fn_f + 1e-5)

        hd_val = compute_hd95_single(filtered_frac, gt_frac_mask)
        hd95_list.append(hd_val)

        # Save diagnostic visualization for sample images
        if i < 15:
            composite = create_diagnostic_composite(raw_bgr, bone_mask, filtered_frac, multiclass_mask, gt_frac_mask)
            out_img_path = os.path.join(args.out_dir, f"multiclass_{base_name}.png")
            cv2.imwrite(out_img_path, composite)

        results.append({
            "image_id": img_name,
            "orig_size": f"{w}x{h}",
            "bone_coverage_pct": round(bone_coverage, 2),
            "raw_fracture_pixels": int(raw_frac_pixels),
            "filtered_fracture_pixels": int(filtered_frac_pixels),
            "fp_filtered_out": int(fp_filtered_out),
            "fracture_dice": round(d_frac, 4),
            "fracture_iou": round(iou_frac, 4),
            "fracture_precision": round(prec_frac, 4),
            "fracture_recall": round(rec_frac, 4),
            "fracture_hd95": round(hd_val, 2),
            "time_sec": round(timing["total_sec"], 3)
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(image_list):
            print(f"[{i+1}/{len(image_list)}] {img_name}: Bone={bone_coverage:.1f}%, FracDice={d_frac:.4f}, FP_Filtered={fp_filtered_out}, Time={timing['total_sec']:.3f}s")

    df_res = pd.DataFrame(results)
    summary_csv = os.path.join(args.out_dir, "hierarchical_cascade_test_summary.csv")
    df_res.to_csv(summary_csv, index=False)

    print("\n" + "=" * 90)
    print("STAGE 5: HIERARCHICAL CASCADE PIPELINE (FLEXRAY + BOUNDARY U-NET 512) - TEST RESULTS")
    print("=" * 90)
    print(f"Mean Fracture Dice (DSC):       {df_res['fracture_dice'].mean():.4f}")
    print(f"Mean Fracture IoU:              {df_res['fracture_iou'].mean():.4f}")
    print(f"Mean Fracture Precision:        {df_res['fracture_precision'].mean():.4f}")
    print(f"Mean Fracture Recall:           {df_res['fracture_recall'].mean():.4f}")
    print(f"Mean Fracture HD95:             {df_res['fracture_hd95'].mean():.2f} px")
    print(f"Total Out-of-Bone FP Removed:   {df_res['fp_filtered_out'].sum():,} pixels")
    print(f"Mean Inference Throughput:      {df_res['time_sec'].mean():.3f} s/image ({1.0/df_res['time_sec'].mean():.1f} FPS)")
    print("=" * 90)


if __name__ == "__main__":
    main()
