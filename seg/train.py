"""
Semantic segmentation model training module.

Implements complete training pipeline for U-Net variants including:
- Multi-architecture support (U-Net, U-Net++, fine-tuned ResNet encoder variants)
- Data augmentation with Albumentations (Optimized for Histology)
- Dice loss optimization with early stopping
- Metrics tracking (Dice coefficient, IoU) with visualization
- Checkpoint management for model persistence

Supports both standard training and demo mode for quick prototyping.
"""

import argparse
import os
import json
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from .segmentation_dataset import SegmentationDataset

plt.switch_backend('Agg')

def set_seed(seed=42):
    """
    Set random seeds for reproducibility across all libraries.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def calculate_metrics(pred, target):
    """
    Calculate Dice coefficient and Intersection over Union (IoU) metrics.
    Expects probabilities (0-1) for pred.
    """
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union_dice = pred.sum() + target.sum()
    union_iou = pred.sum() + target.sum() - intersection
    dice = (2. * intersection) / (union_dice + 1e-6)
    iou = (intersection) / (union_iou + 1e-6)
    return dice.item(), iou.item()


def train_segmentation(train_dir, val_dir, out_dir, model_arch="unet", save_name="best.ckpt", demo=False, epochs=None, batch_size=8, patience=15):
    """
    Train a semantic segmentation model with early stopping.
    Includes auto-fixing for mask values (0-255 -> 0-1) and dimensions.
    """
    set_seed()
    
    os.makedirs(out_dir, exist_ok=True)
    visuals_dir = os.path.join(out_dir, "visuals")
    pred_test_dir = os.path.join(out_dir, "pred_test")
    os.makedirs(visuals_dir, exist_ok=True)
    os.makedirs(pred_test_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if demo:
        n_epochs = 2
        bs = 2 
    else:
        n_epochs = epochs if epochs else 50
        bs = batch_size

    print(f"Config: Arch={model_arch}, Epochs={n_epochs}, Batch={bs}, Patience={patience}, Device={device}")

    # --- MODEL DEFINITION ---
    if model_arch == "finetune":
        model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)
    elif model_arch == "custom":
        model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)
    else:
        model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)

    model.to(device)

    # --- AUGMENTATIONS (Optimized for Histology) ---
    # Histology images have no orientation (cells are rotatable) and stain color varies.
    train_transform = A.Compose([
        A.Resize(512, 512),              # Safety resize
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),           # Vertical flip is valid in histology
        A.RandomRotate90(p=0.5),         # 90 degree rotations
        
        # Color augmentation to handle stain variations (H&E differences)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
        ], p=0.4),
        
        A.Normalize(),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(),
        ToTensorV2()
    ])

    train_ds = SegmentationDataset(os.path.join(train_dir, "images"), os.path.join(train_dir, "masks"), transform=train_transform)
    val_ds = SegmentationDataset(os.path.join(val_dir, "images"), os.path.join(val_dir, "masks"), transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    # CRITICAL FIX 1: from_logits=True ensures numerical stability
    criterion = smp.losses.DiceLoss(mode="binary", from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    best_iou = 0.0
    metrics_history = []
    epochs_no_improve = 0

    print("Starting training...")

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        
        for img, mask in train_loader:
            img = img.to(device)
            mask = mask.to(device)
            
            # --- CRITICAL FIX 2 & 3: Mask Dimensions and Values ---
            # 1. Fix Dimension: Ensure mask is [Batch, 1, H, W]
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            
            # 2. Fix Values: Normalize [0, 255] -> [0, 1]
            if mask.max() > 1:
                mask = mask.float() / 255.0
            else:
                mask = mask.float()
            # ------------------------------------------------------

            optimizer.zero_grad()
            pred = model(img) # Output is Logits
            loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION LOOP ---
        model.eval()
        val_dice_sum = 0
        val_iou_sum = 0
        
        with torch.no_grad():
            for img, mask in val_loader:
                img = img.to(device)
                mask = mask.to(device)

                # Apply same fixes to validation masks
                if mask.ndim == 3:
                    mask = mask.unsqueeze(1)
                if mask.max() > 1:
                    mask = mask.float() / 255.0
                else:
                    mask = mask.float()

                pred_logits = model(img)
                pred_probs = pred_logits.sigmoid() # Convert Logits -> Probabilities for metrics
                
                d, i = calculate_metrics(pred_probs, mask)
                val_dice_sum += d
                val_iou_sum += i
        
        avg_val_dice = val_dice_sum / len(val_loader)
        avg_val_iou = val_iou_sum / len(val_loader)

        print(f"Epoch {epoch+1}/{n_epochs}: Loss={avg_train_loss:.4f} | Val Dice={avg_val_dice:.4f} | Val IoU={avg_val_iou:.4f}")

        metrics_history.append({
            "epoch": epoch + 1,
            "loss": avg_train_loss,
            "dice": avg_val_dice,
            "iou": avg_val_iou
        })

        # Save Best Model
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            epochs_no_improve = 0
            
            final_filename = save_name if save_name.endswith(".ckpt") else f"{save_name}.ckpt"
            torch.save(model.state_dict(), os.path.join(out_dir, final_filename))
            # print(f"  -> Model saved! New Best IoU: {best_iou:.4f}") # Optional verbose
        else:
            epochs_no_improve += 1
            # print(f"  -> No improvement. Patience: {epochs_no_improve}/{patience}")

        # Save history every epoch just in case
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(metrics_history, f)
        
        if epochs_no_improve >= patience:
            print(f"\n[Early Stopping] Training stopped. No improvement in {patience} epochs.")
            break

    # --- POST TRAINING ---
    final_metrics = {
        "best_val_iou": best_iou,
        "final_train_loss": avg_train_loss,
        "epochs_trained": epoch + 1,
        "architecture": model_arch
    }
    with open(os.path.join(out_dir, "val_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=4)
    
    try:
        epochs_range = [x['epoch'] for x in metrics_history]
        loss = [x['loss'] for x in metrics_history]
        dice = [x['dice'] for x in metrics_history]
        iou = [x['iou'] for x in metrics_history]

        plt.figure(figsize=(10, 5))
        plt.plot(epochs_range, loss, label='Train Loss', color='red', linestyle='--')
        plt.plot(epochs_range, dice, label='Val Dice', color='blue')
        plt.plot(epochs_range, iou, label='Val IoU', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.title(f'Training Results: {save_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(visuals_dir, "training_plot.png"))
        plt.close()
        print(f"Plot saved to {os.path.join(visuals_dir, 'training_plot.png')}")
    except Exception as e:
        print(f"Error generating plot: {e}")

    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train semantic segmentation models with configurable architectures and early stopping."
    )
    parser.add_argument("--train_dir", required=True, help="Training data directory with 'images' and 'masks' subdirectories.")
    parser.add_argument("--val_dir", required=True, help="Validation data directory with 'images' and 'masks' subdirectories.")
    parser.add_argument("--out", required=True, help="Output directory for checkpoints and metrics.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and validation (default: 8).")
    parser.add_argument("--patience", type=int, default=15, help="Epochs without improvement before early stopping (default: 15).")
    parser.add_argument("--demo", action="store_true", help="Run quick demo: 2 epochs, batch size 2.")
    parser.add_argument("--model", type=str, default="unet", help="Architecture: 'unet', 'finetune', or 'custom' (default: unet).")
    parser.add_argument("--save_name", type=str, default="best.ckpt", help="Checkpoint filename (default: best.ckpt).")
    
    args = parser.parse_args()
    
    train_segmentation(
        args.train_dir, args.val_dir, args.out, 
        model_arch=args.model, save_name=args.save_name,
        demo=args.demo, epochs=args.epochs, batch_size=args.batch_size,
        patience=args.patience
    )