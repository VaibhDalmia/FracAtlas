import os
import sys
import json
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import torch

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    sam_model_registry = None
    SamPredictor = None


def rasterize_coco_polygons(annotations, h, w):
    """
    Rasterizes a list of COCO segmentation polygons into a single binary mask (uint8).
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in annotations:
        seg_list = ann.get("segmentation", [])
        if isinstance(seg_list, list):
            for seg in seg_list:
                pts = np.array(seg, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(mask, [pts], color=1)
    return mask


def compute_per_image_thresholds(image_gray, extreme_white_thresh=245):
    """
    Computes image-specific relative thresholds via 3-class Multi-Otsu on valid pixels
    [0, extreme_white_thresh], automatically adapting to under/over-exposed X-rays.
    Returns:
      t_air: boundary between detector background air and patient soft tissue
      t_bone: boundary between soft tissue and dense cortical/trabecular bone
    """
    t_air, _ = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    body = image_gray[(image_gray > t_air) & (image_gray <= extreme_white_thresh)]
    if len(body) > 0:
        t_bone, _ = cv2.threshold(body, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        t_bone = t_air + 50
    return float(t_air), float(t_bone)


def segment_edge_polygons(
    image_gray,
    t_air,
    t_bone,
    step=24,
    extreme_white_thresh=245,
):
    """
    Edge-First Polygon Segmentation Pipeline:
      1. Proper Edge Detection: Bilateral noise filtering + CLAHE + Scharr gradient magnitude.
      2. Edge-Guided Polygon Tessellation: Grid-seeded Watershed along gradient ridges.
         Partitions the entire canvas into closed, non-overlapping anatomical polygons bounded by edges.
      3. Complete Polygon Classification:
         - Saturated Outlier (> extreme_white_thresh): Foreign text stamp / metal -> Class 0
         - High Intensity (>= t_bone): Cortical/Trabecular Bone -> Class 2
         - Medium Intensity (t_air < mean < t_bone): Soft Tissue -> Class 1
         - Low Intensity (<= t_air): Background Air -> Class 0
    """
    h, w = image_gray.shape

    # 1. Proper Edge Detection
    filtered = cv2.bilateralFilter(image_gray, 7, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)

    gx = cv2.Scharr(enhanced, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(enhanced, cv2.CV_32F, 0, 1)
    edge_map = cv2.magnitude(gx, gy)
    max_g = np.max(edge_map)
    if max_g > 0:
        edge_map = np.uint8(np.clip(edge_map / max_g * 255.0, 0, 255))
    else:
        edge_map = np.zeros((h, w), dtype=np.uint8)

    # 2. Polygon Seed Placement & Watershed Flooding along Edge Landscape
    grid_y, grid_x = np.mgrid[step // 2 : h : step, step // 2 : w : step]
    seeds_y = grid_y.ravel()
    seeds_x = grid_x.ravel()
    num_polys = len(seeds_y)

    markers = np.zeros((h, w), dtype=np.int32)
    for i, (sy, sx) in enumerate(zip(seeds_y, seeds_x), start=1):
        markers[sy, sx] = i

    cv2.watershed(cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR), markers)

    # 3. Fast Vectorized Polygon Mean Calculation
    valid_m = markers[markers > 0]
    valid_pixels = image_gray[markers > 0]

    sums = np.bincount(valid_m, weights=valid_pixels, minlength=num_polys + 1)
    counts = np.bincount(valid_m, minlength=num_polys + 1)
    means = np.zeros(num_polys + 1)
    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero]

    # 4. Classify each complete polygon
    # Bone: polygons with mean >= t_bone
    bone_ids = np.where((means >= t_bone) & (means <= extreme_white_thresh))[0]
    bone_mask = np.isin(markers, bone_ids).astype(np.uint8)

    # Fill internal trabecular bone marrow cavities to ensure solid anatomical bones
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_CLOSE, kernel)

    # Strict clamping: eliminate text/metal from bone
    bone_mask[image_gray > extreme_white_thresh] = 0
    bone_mask[image_gray <= t_air * 0.7] = 0

    # Tissue: polygons with mean between t_air and t_bone
    tissue_ids = np.where((means > t_air) & (means < t_bone))[0]
    tissue_mask = np.isin(markers, tissue_ids).astype(np.uint8)

    # Exclude bone and text from tissue
    tissue_mask[bone_mask == 1] = 0
    tissue_mask[image_gray > extreme_white_thresh] = 0
    tissue_mask[image_gray <= t_air * 0.7] = 0

    return bone_mask, tissue_mask


def generate_multiclass_masks(
    images_dir="images/Fractured",
    coco_json_path="Annotations/COCO JSON/COCO_fracture_masks.json",
    output_dir="Processed_MultiClass",
    method="edge_polygon",
    sam_checkpoint_path="sam_vit_b_01ec64.pth",
    model_type="vit_b",
    step=24,
    extreme_white_thresh=245,
    num_samples=None,
    save_visualizations=True,
    device=None,
):
    """
    Generates unified 4-class semantic segmentation masks for FracAtlas:
      Class 0: Background (Detector air, text labels, and metallic markers)
      Class 1: Soft Tissue (Patient body / musculature)
      Class 2: Bone (Intact cortical & trabecular skeleton)
      Class 3: Fracture (Ground-truth fracture lesion from COCO annotations)
    """
    bone_out_dir = os.path.join(output_dir, "bone_masks")
    mask_out_dir = os.path.join(output_dir, "multiclass_masks")
    vis_out_dir = os.path.join(output_dir, "visualizations")

    os.makedirs(bone_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)
    if save_visualizations:
        os.makedirs(vis_out_dir, exist_ok=True)

    # 1. Load COCO fracture annotations
    filename_to_anns = {}
    if os.path.exists(coco_json_path):
        print(f"Loading COCO annotations from: {coco_json_path}")
        with open(coco_json_path, "r") as f:
            coco_data = json.load(f)
        id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
        for ann in coco_data["annotations"]:
            fname = id_to_filename.get(ann["image_id"])
            if fname:
                filename_to_anns.setdefault(fname, []).append(ann)
        print(f"Loaded fracture annotations for {len(filename_to_anns)} images.")
    else:
        print(f"Warning: COCO JSON not found at {coco_json_path}. Fractures will default to empty.")

    # 2. Collect image paths
    image_entries = []
    if os.path.isdir(images_dir):
        for root, _, files in os.walk(images_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    image_entries.append((f, os.path.join(root, f)))
    else:
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if num_samples is not None and num_samples > 0:
        image_entries = image_entries[:num_samples]

    print(f"Processing {len(image_entries)} images (Method: {method}, 4-Class Mode)...")

    for img_name, img_path in tqdm(image_entries, desc="Generating Masks"):
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue
        h, w = image_bgr.shape[:2]
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # -------------------------------------------------------------
        # Step A: Rasterize Ground-Truth Fracture Mask from COCO Polygons
        # -------------------------------------------------------------
        frac_mask = np.zeros((h, w), dtype=np.uint8)
        if img_name in filename_to_anns:
            frac_mask = rasterize_coco_polygons(filename_to_anns[img_name], h, w)

        # -------------------------------------------------------------
        # Step B: Image-Specific Relative Thresholds
        # -------------------------------------------------------------
        t_air, t_bone = compute_per_image_thresholds(image_gray, extreme_white_thresh=extreme_white_thresh)

        # -------------------------------------------------------------
        # Step C: Edge Detection + Polygon Classification
        # -------------------------------------------------------------
        bone_mask, tissue_mask = segment_edge_polygons(
            image_gray, t_air, t_bone, step=step, extreme_white_thresh=extreme_white_thresh
        )

        # -------------------------------------------------------------
        # Step D: Construct Unified 4-Class Semantic Mask
        # Class 0: Background (Air, Text labels, Metallic markers)
        # Class 1: Soft Tissue (Patient body excluding bone)
        # Class 2: Bone (Intact bone)
        # Class 3: Fracture (Takes highest precedence)
        # -------------------------------------------------------------
        multiclass_mask = np.zeros((h, w), dtype=np.uint8)

        # Assign Soft Tissue (Class 1)
        multiclass_mask[tissue_mask == 1] = 1

        # Assign Bone (Class 2), overwriting tissue
        multiclass_mask[bone_mask == 1] = 2

        # Enforce Air and Text/Metal remain Background (Class 0)
        multiclass_mask[image_gray <= t_air * 0.7] = 0
        multiclass_mask[image_gray > extreme_white_thresh] = 0

        # Assign Fracture (Class 3), overwriting bone/tissue
        multiclass_mask[frac_mask == 1] = 3

        # -------------------------------------------------------------
        # Step E: Save Lossless PNG Outputs
        # -------------------------------------------------------------
        base_name = os.path.splitext(img_name)[0] + ".png"

        # 1. Binary bone mask (0 or 255)
        cv2.imwrite(os.path.join(bone_out_dir, base_name), bone_mask * 255)

        # 2. Raw semantic mask (0, 1, 2, 3) for PyTorch training
        cv2.imwrite(os.path.join(mask_out_dir, base_name), multiclass_mask)

        # 3. Visual color overlay:
        #    Class 0: Natural X-ray background
        #    Class 1: Cyan / Blue for Soft Tissue
        #    Class 2: Bright Green for Bone
        #    Class 3: Bright Red for Fracture
        if save_visualizations:
            color_mask = np.zeros((h, w, 3), dtype=np.uint8)
            color_mask[multiclass_mask == 1] = [200, 140, 30]   # Cyan/Steel Blue for Soft Tissue
            color_mask[multiclass_mask == 2] = [0, 220, 0]      # Green for Bone
            color_mask[multiclass_mask == 3] = [0, 0, 255]      # Red for Fracture

            overlay = cv2.addWeighted(image_bgr, 0.65, color_mask, 0.35, 0)
            cv2.imwrite(os.path.join(vis_out_dir, base_name), overlay)

    print(f"\nProcessing complete!")
    print(f"  - Bone masks saved to: {bone_out_dir}")
    print(f"  - 4-Class masks (0:BG, 1:Tissue, 2:Bone, 3:Fracture) saved to: {mask_out_dir}")
    if save_visualizations:
        print(f"  - Visual color overlays saved to: {vis_out_dir}")


def main():
    parser = argparse.ArgumentParser(description="FracAtlas Edge-First Polygon Segmentation Mask Generator")
    parser.add_argument("--images_dir", type=str, default="images/Fractured",
                        help="Path to input images directory (default: images/Fractured)")
    parser.add_argument("--coco_json", type=str, default="Annotations/COCO JSON/COCO_fracture_masks.json",
                        help="Path to COCO fracture annotations JSON")
    parser.add_argument("--output_dir", type=str, default="Processed_MultiClass",
                        help="Directory to save output masks")
    parser.add_argument("--method", type=str, default="edge_polygon", choices=["edge_polygon", "sam"],
                        help="Segmentation method: edge_polygon (fast, edge-bounded) or sam")
    parser.add_argument("--step", type=int, default=24,
                        help="Grid polygon step size in pixels (default: 24)")
    parser.add_argument("--extreme_white_thresh", type=int, default=245,
                        help="Threshold above which pixels are treated as foreign text/metal (default: 245)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit processing to first N samples for testing")
    parser.add_argument("--no_vis", action="store_true",
                        help="Disable saving color visual overlay images")
    args = parser.parse_args()

    generate_multiclass_masks(
        images_dir=args.images_dir,
        coco_json_path=args.coco_json,
        output_dir=args.output_dir,
        method=args.method,
        step=args.step,
        extreme_white_thresh=args.extreme_white_thresh,
        num_samples=args.num_samples,
        save_visualizations=not args.no_vis,
    )


if __name__ == "__main__":
    main()
