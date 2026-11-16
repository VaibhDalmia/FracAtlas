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
    Focal Tversky Loss for Highly Imbalanced Medical Segmentation (Abraham & Khan, 2019):
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

        # True Positives, False Positives, False Negatives
        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1.0 - targets_flat)).sum()
        fn = ((1.0 - probs_flat) * targets_flat).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = torch.pow((1.0 - tversky), 1.0 / self.gamma)
        return focal_tversky


def compute_binary_metrics(preds, targets, threshold=0.5, eps=1e-5):
    """
    Computes DSC, IoU, Precision, Recall, and Accuracy for binary fracture masks.
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


def train_epoch(model, loader, optimizer, criterion, device, accum_steps=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (images, masks) in enumerate(tqdm(loader, desc="Training Attention U-Net", leave=False)):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        loss_scaled = loss / accum_steps
        loss_scaled.backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

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
def save_predictions(model, dataset, device, out_dir="results/attention_predictions", num_samples=10, threshold=0.5):
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
        # Green = Ground Truth, Red = Predicted, Yellow = Overlap
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
        save_path = os.path.join(out_dir, f"att_{os.path.splitext(img_name)[0]}.png")
        cv2.imwrite(save_path, comparison)

    print(f"Saved {len(indices)} visual prediction samples to: {out_dir}")


def update_progress_file(epoch, epochs, train_loss, val_metrics, best_val_dice, log_records, log_file="training_log_attention.csv", tag="[Attention U-Net]"):
    # 1. Update CSV
    df = pd.DataFrame(log_records)
    df.to_csv(log_file, index=False)

    # 2. Append progress to PROJECT_PROGRESS.md
    status_md = f"""\n### {tag} Epoch {epoch:02d} / {epochs:02d} Progress Update
- **Train Loss (Focal Tversky):** `{train_loss:.4f}`
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
    parser = argparse.ArgumentParser(description="FracAtlas Attention U-Net with Focal Tversky Loss")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=256, help="Input image dimension (default: 256)")
    parser.add_argument("--base_features", type=int, default=32, help="Base feature channels (default: 32)")
    parser.add_argument("--alpha", type=float, default=0.7, help="Tversky FP penalty weight (default: 0.7)")
    parser.add_argument("--beta", type=float, default=0.3, help="Tversky FN penalty weight (default: 0.3)")
    parser.add_argument("--gamma", type=float, default=1.33, help="Focal exponent (default: 1.33)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--evaluate_only", action="store_true", help="Only run evaluation on test set")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint for evaluation")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'})")

    # Configurable file names based on img_size
    is_hires = args.img_size != 256
    model_filename = f"best_attention_unet_{args.img_size}.pth" if is_hires else "best_attention_unet.pth"
    best_model_path = os.path.join(args.checkpoint_dir, model_filename)
    log_file = f"training_log_attention_{args.img_size}.csv" if is_hires else "training_log_attention.csv"
    pred_dir = f"results/attention_predictions_{args.img_size}" if is_hires else "results/attention_predictions"
    tag = f"[Attention U-Net {args.img_size}x{args.img_size}]" if is_hires else "[Attention U-Net]"
    header = f"\n\n## 8. High-Resolution ({args.img_size}x{args.img_size}) Attention U-Net Experimentation Logs\n" if is_hires else "\n\n## 7. Attention U-Net + Focal Tversky Loss Experimentation Logs\n"

    # 1. Datasets & Loaders
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

    # 2. Model & Optimization
    model = AttentionUNet(in_channels=1, out_channels=1, base_features=args.base_features).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Attention U-Net initialized with {total_params:,} trainable parameters.")

    criterion = FocalTverskyLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    log_records = []

    # Add section header to progress file if training
    if not args.evaluate_only:
        with open("PROJECT_PROGRESS.md", "a", encoding="utf-8") as f:
            f.write(header)

        print(f"\nStarting Attention U-Net Training ({args.img_size}x{args.img_size}, Loss: Focal Tversky alpha={args.alpha}, beta={args.beta}, gamma={args.gamma})...")
        best_val_dice = 0.0

        for epoch in range(1, args.epochs + 1):
            t_start = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, accum_steps=args.grad_accum)
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
                print(f"  --> Saved new best Attention U-Net checkpoint! (Val Dice: {val_dice:.4f})")

            update_progress_file(epoch, args.epochs, train_loss, val_metrics, best_val_dice, log_records, log_file=log_file, tag=tag)

        print(f"\nAttention U-Net Training Complete! Best Validation Dice: {best_val_dice:.4f}")

    # 3. Test Evaluation
    eval_checkpoint = args.checkpoint_path if args.checkpoint_path else best_model_path
    if os.path.exists(eval_checkpoint):
        print(f"\nLoading best weights from: {eval_checkpoint} for final Test evaluation...")
        checkpoint = torch.load(eval_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    print(f"\nEvaluating Attention U-Net ({args.img_size}x{args.img_size}) on Test Set (63 unseen fractured X-rays)...")
    test_metrics = evaluate(model, test_loader, criterion, device)

    print("=" * 65)
    print(f"FINAL TEST SET METRICS (Attention U-Net {args.img_size}x{args.img_size} + Focal Tversky Loss):")
    print(f"  • Dice Similarity Coefficient (DSC): {test_metrics['Dice']:.4f}")
    print(f"  • Intersection over Union (IoU):     {test_metrics['IoU']:.4f}")
    print(f"  • Precision:                         {test_metrics['Precision']:.4f}")
    print(f"  • Recall (Sensitivity):              {test_metrics['Recall']:.4f}")
    print(f"  • Pixel Accuracy:                    {test_metrics['Accuracy']:.4f}")
    print("=" * 65)

    if is_hires:
        test_summary = f"""\n### Final Test Evaluation Results (Attention U-Net {args.img_size}x{args.img_size} vs 256x256 vs Baseline)
| Metric | Baseline U-Net (256x256) | Attention U-Net (256x256) | Attention U-Net ({args.img_size}x{args.img_size}) | Delta vs 256x256 |
| :--- | :---: | :---: | :---: | :---: |
| **Dice Similarity Coefficient (DSC)** | `0.1724` | `0.1892` | **`{test_metrics['Dice']:.4f}`** | **`{test_metrics['Dice'] - 0.1892:+.4f}`** |
| **Intersection over Union (IoU)** | `0.0943` | `0.1045` | **`{test_metrics['IoU']:.4f}`** | **`{test_metrics['IoU'] - 0.1045:+.4f}`** |
| **Precision** | `0.1092` | `0.1597` | **`{test_metrics['Precision']:.4f}`** | **`{test_metrics['Precision'] - 0.1597:+.4f}`** |
| **Recall (Sensitivity)** | `0.4085` | `0.2320` | **`{test_metrics['Recall']:.4f}`** | **`{test_metrics['Recall'] - 0.2320:+.4f}`** |
| **Pixel Accuracy** | `0.9863` | `0.9930` | **`{test_metrics['Accuracy']:.4f}`** | **`{test_metrics['Accuracy'] - 0.9930:+.4f}`** |
"""
    else:
        test_summary = f"""\n### Final Test Evaluation Results (Attention U-Net + Focal Tversky Loss)
| Metric | Baseline U-Net | Attention U-Net | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Dice Similarity Coefficient (DSC)** | `0.1724` | **`{test_metrics['Dice']:.4f}`** | **`{test_metrics['Dice'] - 0.1724:+.4f}`** |
| **Intersection over Union (IoU)** | `0.0943` | **`{test_metrics['IoU']:.4f}`** | **`{test_metrics['IoU'] - 0.0943:+.4f}`** |
| **Precision** | `0.1092` | **`{test_metrics['Precision']:.4f}`** | **`{test_metrics['Precision'] - 0.1092:+.4f}`** |
| **Recall (Sensitivity)** | `0.4085` | **`{test_metrics['Recall']:.4f}`** | **`{test_metrics['Recall'] - 0.4085:+.4f}`** |
| **Pixel Accuracy** | `0.9863` | **`{test_metrics['Accuracy']:.4f}`** | **`{test_metrics['Accuracy'] - 0.9863:+.4f}`** |
"""
    with open("PROJECT_PROGRESS.md", "a", encoding="utf-8") as f:
        f.write(test_summary)

    # 4. Save Visual Predictions
    save_predictions(model, test_dataset, device, out_dir=pred_dir, num_samples=10)


if __name__ == "__main__":
    main()
