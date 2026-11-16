"""
Stage 3: High-Resolution (512x512) 3-Class Attention U-Net on FracAtlas
Evaluates Background (0), Intact Bone (1), and Fracture (2).

Key Innovations over Stage 2:
1. 4x Spatial Pixel Density (512x512 vs 256x256) to resolve thin, hairline cortical fissures (1-3 pixels wide).
2. Gradient Accumulation (batch_size=2, grad_accum=2 -> effective batch size 4)
   to ensure stable convergence while maintaining <2.5 GB peak VRAM on the 6 GB RTX 4050 GPU.
3. Multi-Class Focal Tversky Loss with spatial attention gates.
"""

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

from fracatlas_3class_dataset import FracAtlas3ClassDataset
from attention_unet import AttentionUNet


class MultiClassFocalTverskyLoss(nn.Module):
    """
    Class-Weighted Multi-Class Focal Tversky Loss.
    Heavily weights the fracture class to overcome extreme class imbalance.
    """
    def __init__(self, num_classes=3, alpha=0.7, beta=0.3, gamma=1.33, class_weights=None, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        if class_weights is None:
            self.class_weights = [0.1, 0.3, 0.6]
        else:
            self.class_weights = class_weights

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1) # [B, C, H, W]
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        loss = 0.0
        for c in range(self.num_classes):
            p = probs[:, c, :, :].reshape(-1)
            t = targets_onehot[:, c, :, :].reshape(-1)

            tp = (p * t).sum()
            fp = (p * (1.0 - t)).sum()
            fn = ((1.0 - p) * t).sum()

            if c == 2:
                tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
                focal_tversky = torch.pow((1.0 - tversky), 1.0 / self.gamma)
            else:
                tversky = (2.0 * tp + self.smooth) / (2.0 * tp + fp + fn + self.smooth)
                focal_tversky = 1.0 - tversky

            loss += self.class_weights[c] * focal_tversky

        return loss


class Stage3CompoundLoss(nn.Module):
    """
    Compound Loss: Class-Weighted CE (0.3) + Multi-Class Focal Tversky (0.7)
    """
    def __init__(self, num_classes=3, ce_weight=0.3, ftl_weight=0.7):
        super().__init__()
        self.ce_weight = ce_weight
        self.ftl_weight = ftl_weight
        ce_weights = torch.tensor([0.2, 1.0, 5.0], dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=ce_weights)
        self.ftl = MultiClassFocalTverskyLoss(num_classes=num_classes)

    def to(self, device):
        self.ce.weight = self.ce.weight.to(device)
        return super().to(device)

    def forward(self, logits, targets):
        return self.ce_weight * self.ce(logits, targets) + self.ftl_weight * self.ftl(logits, targets)


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


def compute_multiclass_metrics(preds, targets, num_classes=3, eps=1e-5, compute_hd=False):
    class_names = ["Background", "Bone", "Fracture"]
    metrics = {}

    dices, ious = [], []
    total_correct = (preds == targets).sum().item()
    total_pixels = targets.numel()
    metrics["Accuracy"] = total_correct / total_pixels

    for c in range(num_classes):
        pred_c = (preds == c)
        targ_c = (targets == c)

        tp = (pred_c & targ_c).sum().item()
        fp = (pred_c & ~targ_c).sum().item()
        fn = (~pred_c & targ_c).sum().item()

        d = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        u = (tp + eps) / (tp + fp + fn + eps)
        prec = (tp + eps) / (tp + fp + eps)
        rec = (tp + eps) / (tp + fn + eps)

        dices.append(d)
        ious.append(u)

        name = class_names[c]
        metrics[f"Dice_{name}"] = d
        metrics[f"IoU_{name}"] = u
        metrics[f"Precision_{name}"] = prec
        metrics[f"Recall_{name}"] = rec

    metrics["Macro_Dice"] = float(np.mean(dices))
    metrics["Macro_IoU"] = float(np.mean(ious))

    # Fracture specific clinical metrics
    metrics["Fracture_Dice"] = metrics["Dice_Fracture"]
    metrics["Fracture_IoU"] = metrics["IoU_Fracture"]
    metrics["Fracture_Precision"] = metrics["Precision_Fracture"]
    metrics["Fracture_Recall"] = metrics["Recall_Fracture"]

    if compute_hd:
        hd95_list = []
        preds_np = (preds == 2).cpu().numpy().astype(np.uint8)
        targs_np = (targets == 2).cpu().numpy().astype(np.uint8)
        for i in range(len(preds_np)):
            hd = compute_hd95_single(preds_np[i], targs_np[i])
            hd95_list.append(hd)
        metrics["Fracture_HD95"] = float(np.mean(hd95_list))

    return metrics


def train_epoch(model, loader, optimizer, criterion, device, accum_steps=2):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (images, masks) in enumerate(tqdm(loader, desc="Training Epoch 512", leave=False)):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        loss = loss / accum_steps
        loss.backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, compute_hd=False):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for images, masks in tqdm(loader, desc="Evaluating 512", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_multiclass_metrics(all_preds, all_targets, compute_hd=compute_hd)
    metrics["Loss"] = total_loss / len(loader)
    return metrics


@torch.no_grad()
def save_visual_predictions(model, dataset, device, out_dir="results/3class_attention_512_predictions", num_samples=10):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    for idx in indices:
        img_tensor, mask_tensor = dataset[idx]
        img_name = dataset.image_ids[idx]

        inp = img_tensor.unsqueeze(0).to(device)
        logits = model(inp)
        pred_map = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

        raw_img = (img_tensor[0].numpy() * 255.0).astype(np.uint8)
        gt_map = mask_tensor.numpy().astype(np.uint8)

        h, w = raw_img.shape
        raw_bgr = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)

        # Ground Truth Color Map
        gt_vis = np.zeros((h, w, 3), dtype=np.uint8)
        gt_vis[gt_map == 1] = [180, 105, 0] # Bone
        gt_vis[gt_map == 2] = [0, 0, 255]   # Fracture

        # Predicted Color Map
        pred_vis = np.zeros((h, w, 3), dtype=np.uint8)
        pred_vis[pred_map == 1] = [180, 105, 0]
        pred_vis[pred_map == 2] = [0, 0, 255]

        # Blended Overlay
        blended = raw_bgr.copy()
        overlay = blended.copy()
        overlay[pred_map == 1] = [200, 150, 0]
        overlay[pred_map == 2] = [0, 0, 255]
        cv2.addWeighted(overlay, 0.5, blended, 0.5, 0, blended)

        # Ground truth fracture contour in green
        if (gt_map == 2).sum() > 0:
            contours, _ = cv2.findContours((gt_map == 2).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blended, contours, -1, (0, 255, 0), 2)

        cv2.putText(raw_bgr, "Input X-ray (512x512)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(gt_vis, "GT: Bone(Cyan) Frac(Red)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(pred_vis, "Pred: Bone(Cyan) Frac(Red)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(blended, "Overlay (GT: Grn)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        comparison = np.hstack([raw_bgr, gt_vis, pred_vis, blended])
        save_p = os.path.join(out_dir, f"3class_512_{os.path.splitext(img_name)[0]}.png")
        cv2.imwrite(save_p, comparison)

    print(f"Saved {len(indices)} visual prediction strips to: {out_dir}")


def update_progress_file(epoch, epochs, train_loss, val_metrics, best_macro_dice, log_records, log_file="training_log_3class_attention_512.csv"):
    df = pd.DataFrame(log_records)
    df.to_csv(log_file, index=False)

    entry = f"""
### [High-Res Attention 3-Class U-Net 512x512] Epoch {epoch:02d} / {epochs:02d} Progress Update
- **Train Loss (CE + FTL):** `{train_loss:.4f}`
- **Validation Loss:** `{val_metrics['Loss']:.4f}`
- **Macro Dice:** `{val_metrics['Macro_Dice']:.4f}` *(Best so far: `{best_macro_dice:.4f}`)*
- **Macro IoU:** `{val_metrics['Macro_IoU']:.4f}`
- **Background Dice / IoU:** `{val_metrics['Dice_Background']:.4f}` / `{val_metrics['IoU_Background']:.4f}`
- **Bone Dice / IoU:** `{val_metrics['Dice_Bone']:.4f}` / `{val_metrics['IoU_Bone']:.4f}`
- **Fracture Dice / IoU:** `{val_metrics['Fracture_Dice']:.4f}` / `{val_metrics['Fracture_IoU']:.4f}`
- **Fracture Precision / Recall:** `{val_metrics['Fracture_Precision']:.4f}` / `{val_metrics['Fracture_Recall']:.4f}`
- **Pixel Accuracy:** `{val_metrics['Accuracy']:.4f}`
"""
    with open("PROJECT_PROGRESS_3CLASS.md", "a", encoding="utf-8") as f:
        f.write(entry)


def main():
    parser = argparse.ArgumentParser(description="Stage 3: 3-Class Attention U-Net Training at 512x512 on FracAtlas")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Mini-batch size")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=512, help="Image resolution")
    parser.add_argument("--base_features", type=int, default=32, help="Base feature channels")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--evaluate_only", action="store_true", help="Only run evaluation on test set")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Checkpoint to evaluate")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'})")

    best_model_path = os.path.join(args.checkpoint_dir, "best_3class_attention_unet_512.pth")
    log_file = "training_log_3class_attention_512.csv"

    effective_batch = args.batch_size * args.grad_accum
    print(f"Loading FracAtlas 3-Class Datasets (Resolution: {args.img_size}x{args.img_size})...")
    print(f"Batch config: batch_size={args.batch_size}, grad_accum={args.grad_accum} -> Effective Batch Size={effective_batch}")

    img_dim = (args.img_size, args.img_size)
    train_dataset = FracAtlas3ClassDataset(split="train", img_size=img_dim, augment=True)
    val_dataset = FracAtlas3ClassDataset(split="valid", img_size=img_dim, augment=False)
    test_dataset = FracAtlas3ClassDataset(split="test", img_size=img_dim, augment=False)

    print(f"Dataset splits: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = AttentionUNet(in_channels=1, out_channels=3, base_features=args.base_features).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"512x512 3-Class Attention U-Net initialized with {total_params:,} trainable parameters.")

    criterion = Stage3CompoundLoss(num_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    log_records = []

    if not args.evaluate_only:
        header = "\n\n## 6. Stage 3: High-Resolution 3-Class Attention U-Net (512x512) Experimentation Logs\n"
        with open("PROJECT_PROGRESS_3CLASS.md", "a", encoding="utf-8") as f:
            f.write(header)

        print(f"\nStarting 512x512 3-Class Attention U-Net Training ({args.img_size}x{args.img_size}, {args.epochs} epochs)...")
        best_macro_dice = 0.0

        for epoch in range(1, args.epochs + 1):
            t_start = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, accum_steps=args.grad_accum)
            val_metrics = evaluate(model, val_loader, criterion, device, compute_hd=False)
            elapsed = time.time() - t_start

            macro_dice = val_metrics["Macro_Dice"]
            scheduler.step(macro_dice)
            curr_lr = optimizer.param_groups[0]["lr"]

            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_metrics["Loss"], 4),
                "val_macro_dice": round(macro_dice, 4),
                "val_macro_iou": round(val_metrics["Macro_IoU"], 4),
                "val_bg_dice": round(val_metrics["Dice_Background"], 4),
                "val_bone_dice": round(val_metrics["Dice_Bone"], 4),
                "val_frac_dice": round(val_metrics["Fracture_Dice"], 4),
                "val_frac_prec": round(val_metrics["Fracture_Precision"], 4),
                "val_frac_rec": round(val_metrics["Fracture_Recall"], 4),
                "val_accuracy": round(val_metrics["Accuracy"], 4),
                "lr": curr_lr,
                "time_sec": round(elapsed, 1)
            }
            log_records.append(record)

            is_best = macro_dice > best_macro_dice
            if is_best:
                best_macro_dice = macro_dice
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_dice": macro_dice,
                    "val_metrics": val_metrics
                }, best_model_path)
                star = " [*BEST*]"
            else:
                star = ""

            print(f"Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] - Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['Loss']:.4f} | Macro DSC: {macro_dice:.4f} | Frac DSC: {val_metrics['Fracture_Dice']:.4f} | Frac Rec: {val_metrics['Fracture_Recall']:.4f}{star}")
            update_progress_file(epoch, args.epochs, train_loss, val_metrics, best_macro_dice, log_records, log_file)

        print(f"\nTraining Complete! Best Validation Macro DSC: {best_macro_dice:.4f}")

    # Load best checkpoint for testing
    ckpt_path = args.checkpoint_path or best_model_path
    print(f"\nLoading best checkpoint for final evaluation on unseen test split ({len(test_dataset)} images): {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Full test evaluation with HD95
    t_test0 = time.time()
    test_metrics = evaluate(model, test_loader, criterion, device, compute_hd=True)
    test_duration = time.time() - t_test0

    print("\n" + "=" * 90)
    print("STAGE 3: HIGH-RES 3-CLASS ATTENTION U-NET (512x512) - TEST SET EVALUATION RESULTS")
    print("=" * 90)
    print(f"Pixel Accuracy:                   {test_metrics['Accuracy']:.4f}")
    print(f"Macro Dice Similarity (DSC):      {test_metrics['Macro_Dice']:.4f}")
    print(f"Macro Intersection over Union:    {test_metrics['Macro_IoU']:.4f}")
    print("-" * 90)
    print("CLASS-WISE PERFORMANCE:")
    print(f"  Class 0 (Background):  DSC = {test_metrics['Dice_Background']:.4f} | IoU = {test_metrics['IoU_Background']:.4f}")
    print(f"  Class 1 (Intact Bone): DSC = {test_metrics['Dice_Bone']:.4f} | IoU = {test_metrics['IoU_Bone']:.4f}")
    print(f"  Class 2 (Fracture):    DSC = {test_metrics['Fracture_Dice']:.4f} | IoU = {test_metrics['IoU_Fracture']:.4f} | Prec = {test_metrics['Fracture_Precision']:.4f} | Rec = {test_metrics['Fracture_Recall']:.4f}")
    print(f"  Fracture 95% HD (HD95): {test_metrics['Fracture_HD95']:.2f} px")
    print("-" * 90)
    print(f"Test Latency: {test_duration/len(test_dataset):.4f} s/image ({len(test_dataset)/test_duration:.1f} FPS)")
    print("=" * 90)

    # Save test metrics
    df_test_metrics = pd.DataFrame([test_metrics])
    df_test_metrics.to_csv("results/3class_attention_512_test_metrics.csv", index=False)

    # Save visual predictions
    save_visual_predictions(model, test_dataset, device, out_dir="results/3class_attention_512_predictions", num_samples=10)

    # Append test results to progress tracker
    test_summary_md = f"""
### Final Unseen Test Evaluation (Stage 3: High-Res 3-Class Attention U-Net 512x512)
- **Pixel Accuracy:** `{test_metrics['Accuracy']:.4f}`
- **Macro Dice:** `{test_metrics['Macro_Dice']:.4f}`
- **Macro IoU:** `{test_metrics['Macro_IoU']:.4f}`
- **Background Dice / IoU:** `{test_metrics['Dice_Background']:.4f}` / `{test_metrics['IoU_Background']:.4f}`
- **Bone Dice / IoU:** `{test_metrics['Dice_Bone']:.4f}` / `{test_metrics['IoU_Bone']:.4f}`
- **Fracture Dice / IoU:** `{test_metrics['Fracture_Dice']:.4f}` / `{test_metrics['Fracture_IoU']:.4f}`
- **Fracture Precision / Recall:** `{test_metrics['Fracture_Precision']:.4f}` / `{test_metrics['Fracture_Recall']:.4f}`
- **Fracture 95% Hausdorff Distance (HD95):** `{test_metrics['Fracture_HD95']:.2f} px`
- **Inference Latency:** `{test_duration/len(test_dataset):.4f} s/image` (`{len(test_dataset)/test_duration:.1f} FPS`)
- **Checkpoints & Predictions:** `checkpoints/best_3class_attention_unet_512.pth`, [`results/3class_attention_512_predictions/`](file:///d:/FracAtlas/results/3class_attention_512_predictions/)
"""
    with open("PROJECT_PROGRESS_3CLASS.md", "a", encoding="utf-8") as f:
        f.write(test_summary_md)


if __name__ == "__main__":
    main()
