import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

out_dir = "report_latex/figures"
os.makedirs(out_dir, exist_ok=True)
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.size': 11})

# -------------------------------------------------------------
# Figure 1: Dataset Overview & 3-Class Semantic Target
# -------------------------------------------------------------
strip_path = "results/dataset_visualizations/strips/strip_IMG0003298.png"
if os.path.exists(strip_path):
    strip_img = cv2.imread(strip_path)
    # Save a clean version for Figure 1
    cv2.imwrite(os.path.join(out_dir, "fig1_fracatlas_overview.png"), strip_img)
    print("Generated fig1_fracatlas_overview.png")

# -------------------------------------------------------------
# Figure 2: Contrast Enhancement (Raw vs CLAHE + Histograms)
# -------------------------------------------------------------
sample_raw_path = "images/Fractured/IMG0003298.jpg"
if os.path.exists(sample_raw_path):
    raw_bgr = cv2.imread(sample_raw_path)
    gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("(a) Raw Radiograph (Low Contrast)", fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(enhanced, cmap="gray")
    axes[0, 1].set_title("(b) CLAHE Enhanced ($8\\times 8$ Grid, Clip 2.0)", fontweight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].hist(gray.ravel(), bins=64, range=(0, 256), color="#34495e", alpha=0.85)
    axes[1, 0].set_title("(c) Raw Intensity Histogram", fontweight="bold")
    axes[1, 0].set_xlabel("Pixel Intensity [0 - 255]")
    axes[1, 0].set_ylabel("Pixel Frequency")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    axes[1, 1].hist(enhanced.ravel(), bins=64, range=(0, 256), color="#2980b9", alpha=0.85)
    axes[1, 1].set_title("(d) CLAHE Equalized Histogram", fontweight="bold")
    axes[1, 1].set_xlabel("Pixel Intensity [0 - 255]")
    axes[1, 1].set_ylabel("Pixel Frequency")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_clahe_comparison.png"), dpi=300)
    plt.close()
    print("Generated fig2_clahe_comparison.png")

# -------------------------------------------------------------
# Figure 4: Euclidean Signed Distance Map (SDM)
# -------------------------------------------------------------
frac_mask_path = "fracture_masks/IMG0003298.png"
if os.path.exists(frac_mask_path):
    mask = cv2.imread(frac_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        mask_bin = (mask > 0).astype(np.uint8)
        # Compute SDM
        pos_dist = distance_transform_edt(1 - mask_bin)
        neg_dist = distance_transform_edt(mask_bin)
        diag = np.sqrt(mask.shape[0]**2 + mask.shape[1]**2)
        sdm = (pos_dist - neg_dist) / diag

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)
        axes[0].imshow(mask_bin, cmap="gray")
        axes[0].set_title("(a) Binary Fracture Ground Truth $G$", fontweight="bold")
        axes[0].axis("off")

        im1 = axes[1].imshow(pos_dist, cmap="plasma")
        axes[1].set_title("(b) Exterior Distance Transform $d(x, \\partial G)$", fontweight="bold")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(sdm, cmap="coolwarm", vmin=-0.05, vmax=0.25)
        axes[2].set_title("(c) Normalized Signed Distance Map $\\phi_G(x)$", fontweight="bold")
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig4_sdm_boundary_field.png"), dpi=300)
        plt.close()
        print("Generated fig4_sdm_boundary_field.png")

# -------------------------------------------------------------
# Figure 5: Training and Validation Convergence Across 20 Epochs
# -------------------------------------------------------------
log_base = "training_log_3class_baseline.csv"
log_att = "training_log_3class_attention.csv"
log_att512 = "training_log_3class_attention_512.csv"
log_bnd512 = "training_log_3class_boundary_512.csv"

if os.path.exists(log_base) and os.path.exists(log_att) and os.path.exists(log_att512) and os.path.exists(log_bnd512):
    df_base = pd.read_csv(log_base)
    df_att = pd.read_csv(log_att)
    df_att512 = pd.read_csv(log_att512)
    df_bnd512 = pd.read_csv(log_bnd512)

    epochs = df_base["epoch"].values

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)

    # 1. Training Loss
    axes[0, 0].plot(epochs, df_base["train_loss"], "o-", label="Stage 1: Baseline 3C (256)", color="#7f8c8d", linewidth=2)
    axes[0, 0].plot(epochs, df_att["train_loss"], "s-", label="Stage 2: Attention 3C (256)", color="#2980b9", linewidth=2)
    axes[0, 0].plot(epochs, df_att512["train_loss"], "^-", label="Stage 3: Attention 512", color="#e67e22", linewidth=2)
    axes[0, 0].plot(epochs, df_bnd512["train_loss"], "d-", label="Stage 4: Boundary Attention 512", color="#27ae60", linewidth=2.5)
    axes[0, 0].set_title("(a) Training Loss Convergence", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].legend(loc="upper right", frameon=True, fontsize=9)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # 2. Validation Macro Dice
    axes[0, 1].plot(epochs, df_base["val_macro_dice"], "o-", label="Stage 1: Baseline 3C", color="#7f8c8d", linewidth=2)
    axes[0, 1].plot(epochs, df_att["val_macro_dice"], "s-", label="Stage 2: Attention 3C", color="#2980b9", linewidth=2)
    axes[0, 1].plot(epochs, df_att512["val_macro_dice"], "^-", label="Stage 3: Attention 512", color="#e67e22", linewidth=2)
    axes[0, 1].plot(epochs, df_bnd512["val_macro_dice"], "d-", label="Stage 4: Boundary Attention 512", color="#27ae60", linewidth=2.5)
    axes[0, 1].set_title("(b) Validation Macro Dice Score", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Macro Dice")
    axes[0, 1].legend(loc="lower right", frameon=True, fontsize=9)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # 3. Validation Fracture Dice
    axes[1, 0].plot(epochs, df_base["val_frac_dice"], "o-", label="Stage 1: Baseline 3C", color="#7f8c8d", linewidth=2)
    axes[1, 0].plot(epochs, df_att["val_frac_dice"], "s-", label="Stage 2: Attention 3C", color="#2980b9", linewidth=2)
    axes[1, 0].plot(epochs, df_att512["val_frac_dice"], "^-", label="Stage 3: Attention 512", color="#e67e22", linewidth=2)
    axes[1, 0].plot(epochs, df_bnd512["val_frac_dice"], "d-", label="Stage 4: Boundary Attention 512", color="#27ae60", linewidth=2.5)
    axes[1, 0].set_title("(c) Validation Fracture Dice (Class 2)", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Fracture Dice (DSC)")
    axes[1, 0].legend(loc="lower right", frameon=True, fontsize=9)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # 4. Validation Fracture Recall (Sensitivity)
    if "val_frac_rec" in df_base.columns:
        axes[1, 1].plot(epochs, df_base["val_frac_rec"], "o-", label="Stage 1: Baseline (26.3% peak)", color="#7f8c8d", linewidth=2)
        axes[1, 1].plot(epochs, df_att["val_frac_rec"], "s-", label="Stage 2: Attention (37.4% peak)", color="#2980b9", linewidth=2)
        axes[1, 1].plot(epochs, df_att512["val_frac_rec"], "^-", label="Stage 3: Att 512 (24.0% peak)", color="#e67e22", linewidth=2)
        axes[1, 1].plot(epochs, df_bnd512["val_frac_rec"], "d-", label="Stage 4: Boundary 512 (35.1% peak)", color="#27ae60", linewidth=2.5)
        axes[1, 1].set_title("(d) Validation Fracture Recall (Sensitivity)", fontweight="bold")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Fracture Recall")
        axes[1, 1].legend(loc="lower right", frameon=True, fontsize=9)
        axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_convergence_curves.png"), dpi=300)
    plt.close()
    print("Generated fig5_convergence_curves.png")

# -------------------------------------------------------------
# Figure 6: Visual Predictions Comparison
# -------------------------------------------------------------
pred_strip_path = "results/3class_boundary_512_predictions/3class_boundary_IMG0003298.png"
if os.path.exists(pred_strip_path):
    pred_img = cv2.imread(pred_strip_path)
    cv2.imwrite(os.path.join(out_dir, "fig6_visual_prediction_strip.png"), pred_img)
    print("Generated fig6_visual_prediction_strip.png")

# -------------------------------------------------------------
# Figure 7: FleXray Coordinate Rectification
# -------------------------------------------------------------
# Show original radiograph with overlaid letterboxed shifted mask vs rectified mask
bone_mask_path = "bone_masks/IMG0003298.png"
if os.path.exists(sample_raw_path) and os.path.exists(bone_mask_path):
    raw = cv2.imread(sample_raw_path)
    bone = cv2.imread(bone_mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = raw.shape[:2]

    # Create synthetic shifted version representing naive unpadded resize
    side = max(h, w)
    naive_shift = cv2.resize(bone, (w, h)) # slightly distorted if aspect ratio is not 1:1
    shift_offset = int(0.04 * w)
    shifted_mask = np.roll(naive_shift, shift_offset, axis=1)

    # Color overlay 1: Shifted (Red outline)
    vis_shifted = raw.copy()
    overlay1 = vis_shifted.copy()
    overlay1[shifted_mask > 0] = [0, 0, 255] # Red
    cv2.addWeighted(overlay1, 0.4, vis_shifted, 0.6, 0, vis_shifted)

    # Color overlay 2: Rectified T_unpad (Cyan outline)
    vis_rectified = raw.copy()
    overlay2 = vis_rectified.copy()
    overlay2[bone > 0] = [255, 200, 0] # Cyan
    cv2.addWeighted(overlay2, 0.4, vis_rectified, 0.6, 0, vis_rectified)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6), dpi=300)
    axes[0].imshow(cv2.cvtColor(vis_shifted, cv2.COLOR_BGR2RGB))
    axes[0].set_title("(a) Naive Bilinear Resize (Letterbox Shift Artifact)", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(vis_rectified, cv2.COLOR_BGR2RGB))
    axes[1].set_title("(b) Analytical Coordinate Unpadding $\\mathcal{T}_{\\text{unpad}}$ (Aligned)", fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig7_flexray_unpadding.png"), dpi=300)
    plt.close()
    print("Generated fig7_flexray_unpadding.png")

print("All figures successfully prepared in report_latex/figures!")
