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
from unet_model import UNet


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for Binary Segmentation
    """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class CombinedBCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice Loss for highly imbalanced fracture segmentation
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def compute_binary_metrics(preds, targets, threshold=0.5, eps=1e-5):
    """
    Computes DSC, IoU, Precision, Recall, and Accuracy for binary masks.
    preds: [B, 1, H, W] probabilities
    targets: [B, 1, H, W] binary ground truth
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

    return {
        "Dice": dice,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
    }


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images, masks in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for images, masks in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(outputs)
        all_preds.append(probs.cpu())
        all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_binary_metrics(all_preds, all_targets, threshold=threshold)
    metrics["Loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def save_predictions(model, dataset, device, out_dir="results/predictions", num_samples=10, threshold=0.5):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    for idx in indices:
        img_tensor, mask_tensor = dataset[idx]
        img_name = dataset.image_ids[idx]

        # Model prediction
        inp = img_tensor.unsqueeze(0).to(device)
        logits = model(inp)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred_bin = (prob > threshold).astype(np.uint8)

        # Convert tensors to uint8 images for visualization
        raw_img = (img_tensor[0].numpy() * 255.0).astype(np.uint8)
        gt_mask = mask_tensor[0].numpy().astype(np.uint8)

        # Create RGB overlays:
        # Green = Ground Truth fracture, Red = Model Predicted fracture, Yellow = True Positive overlap
        h, w = raw_img.shape
        vis = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)

        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[gt_mask == 1] = [0, 255, 0]          # Green for GT
        overlay[pred_bin == 1] = [0, 0, 255]         # Red for Pred
        overlay[(gt_mask == 1) & (pred_bin == 1)] = [0, 255, 255]  # Yellow for Overlap

        alpha = 0.4
        blended = cv2.addWeighted(vis, 1.0 - alpha, overlay, alpha, 0)

        # Save side-by-side comparison: Original | Ground Truth | Prediction | Overlay
        gt_vis = cv2.cvtColor(gt_mask * 255, cv2.COLOR_GRAY2BGR)
        pred_vis = cv2.cvtColor(pred_bin * 255, cv2.COLOR_GRAY2BGR)

        comparison = np.hstack([vis, gt_vis, pred_vis, blended])
        save_path = os.path.join(out_dir, f"sample_{os.path.splitext(img_name)[0]}.png")
        cv2.imwrite(save_path, comparison)

    print(f"Saved {len(indices)} visual prediction samples to: {out_dir}")


def update_progress_file(epoch, epochs, train_loss, val_metrics, best_val_dice, log_records):
    """
    Appends progress to PROJECT_PROGRESS.md and training_log.csv
    """
    # 1. Update CSV
    df = pd.DataFrame(log_records)
    df.to_csv("training_log.csv", index=False)

    # 2. Append/Update Markdown summary
    status_md = f"""\n### Epoch {epoch:02d} / {epochs:02d} Progress Update
- **Train Loss:** `{train_loss:.4f}`
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
    parser = argparse.ArgumentParser(description="FracAtlas Binary Fracture U-Net Baseline")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=256, help="Input image dimension (default: 256)")
    parser.add_argument("--base_features", type=int, default=32, help="U-Net base feature channels (default: 32)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--evaluate_only", action="store_true", help="Only run evaluation on test set")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint for evaluation")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'})")

    # 1. Datasets & Loaders
    print("Loading FracAtlas Fracture Datasets...")
    img_dim = (args.img_size, args.img_size)
    train_dataset = FracAtlasFractureDataset(split="train", img_size=img_dim, augment=True)
    val_dataset = FracAtlasFractureDataset(split="valid", img_size=img_dim, augment=False)
    test_dataset = FracAtlasFractureDataset(split="test", img_size=img_dim, augment=False)

    print(f"Dataset splits: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. Model & Optimization
    model = UNet(in_channels=1, out_channels=1, base_features=args.base_features).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"U-Net initialized with {total_params:,} trainable parameters.")

    criterion = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_model_path = os.path.join(args.checkpoint_dir, "best_unet.pth")
    log_records = []

    if not args.evaluate_only:
        print("\nStarting U-Net Training...")
        best_val_dice = 0.0

        for epoch in range(1, args.epochs + 1):
            t_start = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
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

            # Save best checkpoint
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                }, best_model_path)
                print(f"  --> Saved new best checkpoint! (Val Dice: {val_dice:.4f})")

            # Update progress file
            update_progress_file(epoch, args.epochs, train_loss, val_metrics, best_val_dice, log_records)

        print(f"\nTraining Complete! Best Validation Dice: {best_val_dice:.4f}")

    # 3. Test Evaluation
    eval_checkpoint = args.checkpoint_path if args.checkpoint_path else best_model_path
    if os.path.exists(eval_checkpoint):
        print(f"\nLoading best weights from: {eval_checkpoint} for final Test evaluation...")
        checkpoint = torch.load(eval_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    print("\nEvaluating on Test Set (63 unseen fractured X-rays)...")
    test_metrics = evaluate(model, test_loader, criterion, device)

    print("=" * 60)
    print("FINAL TEST SET METRICS (FracAtlas U-Net Baseline):")
    print(f"  • Dice Similarity Coefficient (DSC): {test_metrics['Dice']:.4f}")
    print(f"  • Intersection over Union (IoU):     {test_metrics['IoU']:.4f}")
    print(f"  • Precision:                         {test_metrics['Precision']:.4f}")
    print(f"  • Recall (Sensitivity):              {test_metrics['Recall']:.4f}")
    print(f"  • Pixel Accuracy:                    {test_metrics['Accuracy']:.4f}")
    print("=" * 60)

    # Append test results to progress file
    test_summary = f"""\n## Final Test Evaluation Results (Unseen 63 Test Images)
| Metric | Score | Clinical Description |
| :--- | :---: | :--- |
| **Dice Similarity Coefficient (DSC)** | **`{test_metrics['Dice']:.4f}`** | Spatial overlap between predicted fracture and ground truth |
| **Intersection over Union (IoU)** | **`{test_metrics['IoU']:.4f}`** | Jaccard index measuring area of overlap over area of union |
| **Precision** | **`{test_metrics['Precision']:.4f}`** | Proportion of predicted fracture pixels that are genuine lesions |
| **Recall (Sensitivity)** | **`{test_metrics['Recall']:.4f}`** | Proportion of true fracture lesions successfully detected |
| **Pixel Accuracy** | **`{test_metrics['Accuracy']:.4f}`** | Overall pixel classification accuracy |
"""
    with open("PROJECT_PROGRESS.md", "a", encoding="utf-8") as f:
        f.write(test_summary)

    # 4. Generate Visual Predictions
    save_predictions(model, test_dataset, device, out_dir="results/predictions", num_samples=10)


if __name__ == "__main__":
    main()
