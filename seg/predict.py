"""
Image segmentation inference module.

Provides functionality for generating semantic segmentation masks on input images
using pre-trained U-Net models. Supports batch processing of directories and 
single image inference with automatic architecture detection.

This module handles:
- Image preprocessing and normalization
- Model inference with GPU/CPU device management
- Mask post-processing and resizing
- Checkpoint loading with automatic architecture detection
"""

import torch
import argparse
import os
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


def predict_image(model, image_path, device):
    """
    Generate semantic segmentation mask for a single image.
    
    Performs image preprocessing, model inference, and post-processing
    to produce a binary segmentation mask resized to the original image dimensions.
    
    Args:
        model: Trained segmentation model in eval mode.
        image_path (str): Path to the input image file.
        device (str): Target device for inference ('cuda' or 'cpu').
    
    Returns:
        PIL.Image or None: Grayscale mask image (0-255) or None if processing fails.
        The mask is resized to match the original image dimensions.
    
    Raises:
        Logs errors for unsupported image formats but returns None instead of raising.
    """
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(),
        ToTensorV2()
    ])
    
    try:
        original_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Skipping file {image_path}: {e}")
        return None

    original_size = original_img.size
    img_np = np.array(original_img)
    augmented = transform(image=img_np)["image"]
    img_tensor = augmented.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(img_tensor)
        pred = pred.sigmoid()
        mask = (pred > 0.5).float().cpu().numpy()[0, 0]
    
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.resize(original_size, Image.NEAREST)
    
    return mask_img

def get_model_structure(arch, device):
    """
    Instantiate segmentation model architecture.
    
    Factory function that creates an uninitialized model with the specified
    architecture. Used for architecture detection during checkpoint loading.
    
    Args:
        arch (str): Architecture identifier. Supported values:
            - 'unet': Standard U-Net with ResNet34 encoder
            - 'finetune': U-Net with ResNet50 encoder for transfer learning
            - 'custom': U-Net++ (UnetPlusPlus) with ResNet34 encoder
        device (str): Target device for model ('cuda' or 'cpu').
    
    Returns:
        torch.nn.Module: Uninitialized segmentation model on the target device.
        Defaults to 'unet' architecture if unsupported arch is provided.
    """
    if arch == "finetune":
        return smp.Unet(encoder_name="resnet50", in_channels=3, classes=1).to(device)
    elif arch == "custom":
        return smp.UnetPlusPlus(encoder_name="resnet34", in_channels=3, classes=1).to(device)
    else:
        return smp.Unet(encoder_name="resnet34", in_channels=3, classes=1).to(device)

def main():
    """
    Main entry point for image segmentation inference.
    
    Orchestrates the inference pipeline:
    1. Parses command-line arguments
    2. Initializes device (GPU/CPU)
    3. Automatically detects and loads model architecture
    4. Processes input (single image or batch directory)
    5. Saves segmentation masks with 'mask_' prefix
    
    Command-line arguments:
        --ckpt: Path to model checkpoint file (required)
        --input: Path to image or directory containing images (required)
        --out: Output directory for segmentation masks (required)
    
    Supported formats: .png, .jpg, .jpeg, .tif, .tiff
    
    Raises:
        SystemExit: If no compatible architecture is found for the checkpoint.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input", required=True, help="Path to image or folder")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    architectures = ["unet", "finetune", "custom"]
    model = None
    
    print(f"Loading checkpoint: {args.ckpt}")
    state_dict = torch.load(args.ckpt, map_location=device)

    for arch in architectures:
        try:
            print(f"Trying to load as '{arch}'...", end=" ")
            temp_model = get_model_structure(arch, device)
            temp_model.load_state_dict(state_dict)
            
            print("SUCCESS!")
            model = temp_model
            model.eval()
            break 
        except RuntimeError:
            print("Failed (Architecture mismatch)")
        except Exception as e:
            print(f"Error: {e}")

    if model is None:
        print("\nCRITICAL ERROR: No matching architecture found.")
        exit(1)
    
    os.makedirs(args.out, exist_ok=True)

    if os.path.isdir(args.input):
        from pathlib import Path
        files = list(Path(args.input).glob("*"))
        print(f"Processing {len(files)} files...")
        for p in files:
            if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                mask = predict_image(model, p, device)
                if mask:
                    output_name = f"mask_{p.name}"
                    mask.save(os.path.join(args.out, output_name))
    else:
        mask = predict_image(model, args.input, device)
        if mask:
            filename = os.path.basename(args.input)
            output_name = f"mask_{filename}"
            save_path = os.path.join(args.out, output_name)
            print(f"Saving mask to: {save_path}")
            mask.save(save_path)
        else:
            print("Error processing image")
            exit(1)

if __name__ == "__main__":
    main()