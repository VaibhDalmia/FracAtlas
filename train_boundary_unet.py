import os
import argparse
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from fracatlas_dataset import FracAtlasFractureDataset
from attention_unet import AttentionUNet


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss (Abraham & Khan, 2019):
    Specifically designed to penalize False Positives (alpha=0.7) while maintaining high Recall (beta=0.3),
    with a focal parameter (gamma=1.33) focusing training on hard fracture boundaries.
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1.0 - targets_flat)).sum()
        fn = ((1.0 - probs_flat) * targets_flat).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = torch.pow((1.0 - tversky), 1.0 / self.gamma)
        return focal_tversky


class BoundaryLoss(nn.Module):
    """
    Boundary Loss (Kervadec et al., MIDL 2019 / IEEE TMI 2021):
    Penalizes predicted probabilities proportionally to their Euclidean distance from the true boundary.
    Uses Signed Distance Maps (SDMs) normalized by the image canvas diagonal.
    """
    def forward(self, logits, sdms):
        probs = torch.sigmoid(logits)
        return torch.mean(probs * sdms)


def compute_sdm_batch(masks):
    """
    Computes normalized Signed Distance Maps (SDMs) for a batch of binary masks:
    Negative inside the ground truth lesion, positive outside.
    """
    B, _, H, W = masks.shape
    sdms = []
    masks_np = (masks > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)
    diag = np.sqrt(H**2 + W**2)

    for b in range(B):
        m = masks_np[b]
        if m.max() == 0:
            sdm = np.ones((H, W), dtype=np.float32)
        else:
            dist_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
            dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 3)
            sdm = (dist_out - dist_in) / diag
        sdms.append(sdm)

    return torch.from_numpy(np.stack(sdms)).unsqueeze(1).to(masks.device, dtype=torch.float32)


def compute_hd95_single(pred_mask, gt_mask):
    """
    Computes the 95th percentile Hausdorff Distance (HD95) between binary masks in pixels.
    """
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


def compute_binary_metrics(preds, targets, threshold=0.5, eps=1e-5, compute_hd=False):
    """
    Computes DSC, IoU, Precision, Recall, Accuracy, and optionally HD95.
    """
    pred_bin = (preds > threshold).float()
    targ_bin = (targets > threshold).float()

    tp = (pred_bin * targ_bin).sum().item()
    fp = (pred_bin * (1.0 - targ_bin)).sum().item()
    fn = ((1.0 - pred_bin) * targ_bin).sum().item()
    tn = ((1.0 - pred_bin) * (1.0 - targ_bin)).sum().item()

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    metrics = {
        "Dice": dice,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
    }

    if compute_hd:
        hd95_list = []
        preds_np = (preds > threshold).squeeze(1).numpy().astype(np.uint8)
        targs_np = (targets > threshold).squeeze(1).numpy().astype(np.uint8)
        for i in range(len(preds_np)):
            hd = compute_hd95_single(preds_np[i], targs_np[i])
            hd95_list.append(hd)
        metrics["HD95"] = float(np.mean(hd95_list))

    return metrics


def train_epoch(model, loader, optimizer, criterion_ftl, criterion_boundary, boundary_weight, device, accum_steps=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (images, masks) in enumerate(tqdm(loader, desc="Training Boundary-Aware U-Net", leave=False)):
        images = images.to(device)
        masks = masks.to(device)
        sdms = compute_sdm_batch(masks)

        outputs = model(images)
        loss_ftl = criterion_ftl(outputs, masks)
        loss_bd = criterion_boundary(outputs, sdms)
        loss = loss_ftl + boundary_weight * loss_bd

        loss_scaled = loss / accum_steps
        loss_scaled.backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion_ftl, criterion_boundary, boundary_weight, device, threshold=0.5, compute_hd=False):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for images, masks in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        sdms = compute_sdm_batch(masks)

        outputs = model(images)
        loss_ftl = criterion_ftl(outputs, masks)
        loss_bd = criterion_boundary(outputs, sdms)
        loss = loss_ftl + boundary_weight * loss_bd
        total_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(outputs)
        all_preds.append(probs.cpu())
        all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_binary_metrics(all_preds, all_targets, threshold=threshold, compute_hd=compute_hd)
    metrics["Loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def save_predictions(model, dataset, device, out_dir="results/boundary_predictions_512", num_samples=10, threshold=0.5):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    for idx in indices:
        img_tensor, mask_tensor = dataset[idx]
        img_name = dataset.image_ids[idx]

        inp = img_tensor.unsqueeze(0).to(device)
        logits = model(inp)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred_bin = (prob > threshold).astype(np.uint8)

        raw_img = (img_tensor[0].numpy() * 255.0).astype(np.uint8)
        gt_mask = mask_tensor[0].numpy().astype(np.uint8)

        h, w = raw_img.shape
        vis = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)

        blended = vis.copy()
        blended[gt_mask == 1] = [0, 255, 0]      # Green = Ground Truth
        blended[pred_bin == 1] = [0, 0, 255]     # Red = Predicted
        blended[(gt_mask == 1) & (pred_bin == 1)] = [0, 255, 255] # Yellow = Overlap

        gt_vis = cv2.cvtColor(gt_mask * 255, cv2.COLOR_GRAY2BGR)
        pred_vis = cv2.cvtColor(pred_bin * 255, cv2.COLOR_GRAY2BGR)

        comparison = np.hstack([vis, gt_vis, pred_vis, blended])
        save_path = os.path.join(out_dir, f"boundary_{os.path.splitext(img_name)[0]}.png")
        cv2.imwrite(save_path, comparison)

    print(f"Saved {len(indices)} visual prediction samples to: {out_dir}")


def update_progress_file(epoch, epochs, train_loss, val_metrics, best_val_dice, log_records, log_file="training_log_boundary_512.csv"):
    df = pd.DataFrame(log_records)
    df.to_csv(log_file, index=False)

    status_md = f"""\n### [Boundary-Aware Attention U-Net 512x512] Epoch {epoch:02d} / {epochs:02d} Progress Update
- **Train Loss (Tversky + Boundary):** `{train_loss:.4f}`
- **Validation Loss:** `{val_metrics['Loss']:.4f}`
- **Validation Dice (DSC):** `{val_metrics['Dice']:.4f}` *(Best so far: `{best_val_dice:.4f}`)*
- **Validation IoU:** `{val_metrics['IoU']:.4f}`
- **Validation Precision:** `{val_metrics['Precision']:.4f}`
- **Validation Recall:** `{val_metrics['Recall']:.4f}`
- **Validation Accuracy:** `{val_metrics['Accuracy']:.4f}`
"""
    with open("PROJECT_PROGRESS.md", "a", encoding="utf-8") as f:
        f.write(status_md)


def main():
    parser = argparse.ArgumentParser(description="FracAtlas Boundary-Aware Attention U-Net (512x512)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=512, help="Input image dimension (default: 512)")
    parser.add_argument("--base_features", type=int, default=32, help="Base feature channels (default: 32)")
    parser.add_argument("--alpha", type=float, default=0.7, help="Tversky FP penalty weight (default: 0.7)")
    parser.add_argument("--beta", type=float, default=0.3, help="Tversky FN penalty weight (default: 0.3)")
    parser.add_argument("--gamma", type=float, default=1.33, help="Focal exponent (default: 1.33)")
    parser.add_argument("--boundary_weight", type=float, default=0.5, help="Weight for Boundary Loss (default: 0.5)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--evaluate_only", action="store_true", help="Only run evaluation on test set")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint for evaluation")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'})")

    best_model_path = os.path.join(args.checkpoint_dir, "best_boundary_unet_512.pth")
    log_file = "training_log_boundary_512.csv"
    pred_dir = "results/boundary_predictions_512"

    print(f"Loading FracAtlas Fracture Datasets (Resolution: {args.img_size}x{args.img_size})...")
    img_dim = (args.img_size, args.img_size)
    train_dataset = FracAtlasFractureDataset(split="train", img_size=img_dim, augment=True)
    val_dataset = FracAtlasFractureDataset(split="valid", img_size=img_dim, augment=False)
    test_dataset = FracAtlasFractureDataset(split="test", img_size=img_dim, augment=False)

    print(f"Dataset splits: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")
    effective_batch = args.batch_size * args.grad_accum
    print(f"Batch config: batch_size={args.batch_size}, grad_accum={args.grad_accum} -> Effective Batch Size={effective_batch}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = AttentionUNet(in_channels=1, out_channels=1, base_features=args.base_features).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Boundary-Aware Attention U-Net initialized with {total_params:,} trainable parameters.")

    criterion_ftl = FocalTverskyLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    criterion_boundary = BoundaryLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    log_records = []

    if not args.evaluate_only:
        header = "\n\n## 9. Boundary-Aware Attention U-Net (512x512) + Boundary Loss Experimentation Logs\n"
        with open("PROJECT_PROGRESS.md", "a", encoding="utf-8") as f:
            f.write(header)

        print(f"\nStarting Boundary-Aware Training ({args.img_size}x{args.img_size}, FTL + Boundary Loss weight={args.boundary_weight})...")
        best_val_dice = 0.0

        for epoch in range(1, args.epochs + 1):
            t_start = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, criterion_ftl, criterion_boundary, args.boundary_weight, device, accum_steps=args.grad_accum)
            val_metrics = evaluate(model, val_loader, criterion_ftl, criterion_boundary, args.boundary_weight, device)
            elapsed = time.time() - t_start

            val_dice = val_metrics["Dice"]
            val_iou = val_metrics["IoU"]
            scheduler.step(val_dice)

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["Loss"],
                "val_dice": val_dice,
                "val_iou": val_iou,
                "val_precision": val_metrics["Precision"],
                "val_recall": val_metrics["Recall"],
                "val_accuracy": val_metrics["Accuracy"],
                "time_sec": elapsed
            }
            log_records.append(record)

            print(f"Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] - "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['Loss']:.4f} | "
                  f"Val Dice: {val_dice:.4f} | "
                  f"Val IoU: {val_iou:.4f} | "
                  f"Val Precision: {val_metrics['Precision']:.4f} | "
                  f"Val Recall: {val_metrics['Recall']:.4f}")

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                }, best_model_path)
                print(f"  --> Saved new best Boundary-Aware checkpoint! (Val Dice: {val_dice:.4f})")

            update_progress_file(epoch, args.epochs, train_loss, val_metrics, best_val_dice, log_records, log_file=log_file)

        print(f"\nBoundary-Aware Training Complete! Best Validation Dice: {best_val_dice:.4f}")

    eval_checkpoint = args.checkpoint_path if args.checkpoint_path else best_model_path
    if os.path.exists(eval_checkpoint):
        print(f"\nLoading best weights from: {eval_checkpoint} for final Test evaluation...")
        checkpoint = torch.load(eval_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    print(f"\nEvaluating Boundary-Aware Attention U-Net on Test Set (63 unseen fractured X-rays) with HD95...")
    test_metrics = evaluate(model, test_loader, criterion_ftl, criterion_boundary, args.boundary_weight, device, compute_hd=True)

    print("=" * 65)
    print(f"FINAL TEST SET METRICS (Boundary-Aware Attention U-Net 512x512):")
    print(f"  • Dice Similarity Coefficient (DSC): {test_metrics['Dice']:.4f}")
    print(f"  • Intersection over Union (IoU):     {test_metrics['IoU']:.4f}")
    print(f"  • Precision:                         {test_metrics['Precision']:.4f}")
    print(f"  • Recall (Sensitivity):              {test_metrics['Recall']:.4f}")
    print(f"  • Pixel Accuracy:                    {test_metrics['Accuracy']:.4f}")
    print(f"  • 95% Hausdorff Distance (HD95):     {test_metrics.get('HD95', 0.0):.2f} px")
    print("=" * 65)

    test_summary = f"""\n### Final Test Evaluation Results (Boundary-Aware Attention U-Net 512x512 vs Prior Stages)
| Metric | Baseline U-Net (256x256) | Attention U-Net (256x256) | Attention U-Net (512x512) | Boundary-Aware U-Net (512x512) |
| :--- | :---: | :---: | :---: | :---: |
| **Dice Similarity Coefficient (DSC)** | `0.1724` | `0.1892` | `0.1968` | **`{test_metrics['Dice']:.4f}`** |
| **Intersection over Union (IoU)** | `0.0943` | `0.1045` | `0.1092` | **`{test_metrics['IoU']:.4f}`** |
| **Precision** | `0.1092` | `0.1597` | `0.1855` | **`{test_metrics['Precision']:.4f}`** |
| **Recall (Sensitivity)** | `0.4085` | `0.2320` | `0.2097` | **`{test_metrics['Recall']:.4f}`** |
| **Pixel Accuracy** | `0.9863` | `0.9930` | `0.9940` | **`{test_metrics['Accuracy']:.4f}`** |
| **95% Hausdorff Distance (HD95)** | *--* | *--* | *--* | **`{test_metrics.get('HD95', 0.0):.2f} px`** |
"""
    with open("PROJECT_PROGRESS.md", "a", encoding="utf-8") as f:
        f.write(test_summary)

    save_predictions(model, test_dataset, device, out_dir=pred_dir, num_samples=10)


if __name__ == "__main__":
    main()
