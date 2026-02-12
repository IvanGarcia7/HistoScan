"""
Whole Slide Image (WSI) segmentation inference module.

Processes large whole slide images by tiling and performing semantic segmentation
on tissue regions. Implements intelligent tissue detection, adaptive region selection,
and multi-tile reconstruction with overlay visualization.

Key features:
- Automatic tissue region detection and center-of-mass calculation
- Smart filtering to identify valid tissue tiles
- Grid-based tiling with configurable overlap
- Automatic model architecture detection from checkpoint
- Full WSI reconstruction with color-coded segmentation overlay
- GPU-accelerated inference with fallback to CPU

Typical workflow:
1. Load WSI slide and model checkpoint
2. Detect tissue regions and find optimal starting point
3. Extract tiles in expanding grid pattern
4. Run inference on valid tiles
5. Reconstruct full image with segmentation mask overlay
"""

import argparse
import os
import torch
import numpy as np
import openslide
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageStat
import math
import sys


def get_model_structure(arch, device):
    """
    Instantiate segmentation model architecture.
    
    Factory function for creating uninitialized segmentation models with different
    architectures. Used in conjunction with automatic architecture detection.
    
    Args:
        arch (str): Architecture identifier. Supported values:
            - 'finetune': U-Net with ResNet50 encoder
            - 'custom': U-Net++ (UnetPlusPlus) with ResNet34 encoder
            - 'unet' or default: Standard U-Net with ResNet34 encoder
        device (str): Target device ('cuda' or 'cpu').
    
    Returns:
        torch.nn.Module: Uninitialized model on the target device.
    """
    import segmentation_models_pytorch as smp
    if arch == "finetune":
        return smp.Unet(encoder_name="resnet50", in_channels=3, classes=1).to(device)
    elif arch == "custom":
        return smp.UnetPlusPlus(encoder_name="resnet34", in_channels=3, classes=1).to(device)
    else:
        return smp.Unet(encoder_name="resnet34", in_channels=3, classes=1).to(device)

def load_model_auto(ckpt_path, device):
    """
    Load model checkpoint with automatic architecture detection.
    
    Attempts to load checkpoint with different model architectures until a compatible
    match is found. Falls back to strict=False loading as final attempt if all
    architecture-specific loads fail.
    
    Args:
        ckpt_path (str): Path to model checkpoint file.
        device (str): Target device for loading ('cuda' or 'cpu').
    
    Returns:
        torch.nn.Module: Loaded model in eval mode, ready for inference.
    
    Raises:
        ValueError: If no compatible architecture is found for the checkpoint.
    """
    print(f"Loading model: {ckpt_path}", flush=True)
    state_dict = torch.load(ckpt_path, map_location=device)
    architectures = ["unet", "finetune", "custom"]
    for arch in architectures:
        try:
            model = get_model_structure(arch, device)
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except:
            continue
    try:
        model = get_model_structure("unet", device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    except:
        raise ValueError("Architecture mismatch")

def is_valid_tile(tile_pil, tissue_percentage=0.01):
    """
    Validate whether a tile contains meaningful tissue content.
    
    Applies heuristic checks to detect valid tissue tiles and filter out artifacts:
    - Rejects tiles with average intensity < 40 (background/dark noise)
    - Rejects tiles with average intensity > 240 (white space/empty)
    - Rejects tiles with low standard deviation < 5 (blank/uniform areas)
    
    Args:
        tile_pil (PIL.Image): Input tile image in RGB or grayscale.
        tissue_percentage (float): Unused parameter (legacy). Default is 0.01.
    
    Returns:
        bool: True if tile appears to contain valid tissue, False otherwise.
    """
    try:
        if tile_pil.size[0] == 0:
            return False
        gray = tile_pil.convert("L")
        np_gray = np.array(gray)
        avg = np.mean(np_gray)
        
        if avg < 40:
            return False
        if avg > 240:
            return False
        
        stat = ImageStat.Stat(gray)
        if stat.stddev[0] < 5:
            return False

        return True
    except:
        return False

def find_starting_point(slide, demo=False):
    """
    Detect tissue region and find optimal starting point for tiling.
    
    Analyzes slide thumbnail to identify tissue boundaries and compute the best
    center point for grid-based tile extraction. Uses two strategies:
    - Normal mode: Computes center-of-mass of tissue region
    - Demo mode: Finds highest-density tissue patch for guaranteed valid tiles
    
    Args:
        slide (openslide.OpenSlide): Opened whole slide image object.
        demo (bool): If True, uses high-contrast point detection instead of
            center-of-mass. Ensures demo runs find valid tissue quickly.
            Default is False.
    
    Returns:
        tuple: (x, y) coordinates of optimal tiling center point in slide space.
    
    Notes:
        - Creates thumbnail with automatic scaling (max 1024px)
        - Tissue detected by thresholding: 40 < intensity < 220
        - Demo mode finds grid cell with maximum tissue concentration
        - Returns slide center if tissue detection fails
    """
    try:
        w, h = slide.dimensions
        scale = max(w, h) // 1024
        if scale == 0:
            scale = 1
        
        thumb = slide.get_thumbnail((w // scale, h // scale)).convert("L")
        np_thumb = np.array(thumb)
        
        mask = (np_thumb > 40) & (np_thumb < 220)
        
        if demo:
            print("  [Demo Mode] Searching for high-contrast tissue point...", flush=True)
            best_x, best_y = w // 2, h // 2
            max_score = -1
            
            step = 50
            h_thumb, w_thumb = np_thumb.shape
            
            for y in range(0, h_thumb, step):
                for x in range(0, w_thumb, step):
                    patch = mask[y:y+step, x:x+step]
                    score = np.sum(patch)
                    if score > max_score:
                        max_score = score
                        best_y = y + step//2
                        best_x = x + step//2
            
            print(f"  [Demo Mode] Hot point found at thumbnail: {best_x}, {best_y}", flush=True)
            return int(best_x * scale), int(best_y * scale)

        else:
            y_idxs, x_idxs = np.nonzero(mask)
            if len(x_idxs) == 0:
                return w // 2, h // 2
            cx = int(np.mean(x_idxs) * scale)
            cy = int(np.mean(y_idxs) * scale)
            return cx, cy

    except Exception as e:
        print(f"Error finding start ({e}), using center.", flush=True)
        return slide.dimensions[0] // 2, slide.dimensions[1] // 2

def main():
    """
    Main entry point for WSI segmentation inference.
    
    Orchestrates the complete pipeline:
    1. Parse command-line arguments and initialize device
    2. Load model checkpoint with automatic architecture detection
    3. Open whole slide image
    4. Detect tissue regions and find optimal starting point
    5. Phase 1 (Smart search): Expand from center in concentric grids to find
       minimum number of valid tiles
    6. Phase 2 (Processing): Extract full tiling grid, run inference on valid
       tiles, and reconstruct full image
    7. Generate overlay visualization with color-coded segmentation
    
    Command-line arguments:
        --slide: Path to WSI file (required)
        --ckpt: Path to model checkpoint (required)
        --out: Output directory for results (required)
        --num_tiles: Target number of tiles to process (default: 10)
        --tile_size: Size of square tiles in pixels (default: 1024)
        --overlap: Pixel overlap between adjacent tiles (default: 64)
        --smart_filter: Enable intelligent tissue detection filtering
        --demo_mode: Use demo mode for quick testing with high-density regions
    
    Outputs:
        - tile_0.jpg, tile_1.jpg, ...: Individual processed tiles with overlay
        - full_reconstruction.jpg: Complete WSI reconstruction with green overlay
          on detected regions
    
    Notes:
        - Requires GPU for efficient processing; falls back to CPU
        - Generates expanding grid search pattern for smart filtering
        - Enforces center tile validity in demo mode for consistent output
        - Memory-intensive for large WSI images; exits gracefully on allocation failure
    """
    parser = argparse.ArgumentParser(
        description="Process whole slide images with semantic segmentation inference."
    )
    parser.add_argument("--slide", required=True, help="Path to whole slide image file.")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint.")
    parser.add_argument("--out", required=True, help="Output directory for reconstructed image and tiles.")
    parser.add_argument("--num_tiles", type=int, default=10, help="Target number of tiles to process (default: 10).")
    parser.add_argument("--tile_size", type=int, default=1024, help="Tile size in pixels (default: 1024).")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap between tiles in pixels (default: 64).")
    parser.add_argument("--smart_filter", action="store_true", help="Enable smart tissue filtering.")
    parser.add_argument("--demo_mode", action="store_true", help="Use demo mode for quick testing.")
    args = parser.parse_args()

    print(f"--- WSI PROCESS STARTED (Demo: {args.demo_mode}) ---", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    try:
        model = load_model_auto(args.ckpt, device)
        slide = openslide.OpenSlide(args.slide)
    except Exception as e:
        print(f"Error setup: {e}", flush=True)
        return

    tile_size = args.tile_size
    stride = tile_size - args.overlap
    
    center_x, center_y = find_starting_point(slide, demo=args.demo_mode)
    
    current_radius = 0
    valid_tiles_found = 0
    MAX_RADIUS = 6 
    
    if args.smart_filter:
        print(f"Scanning area around ({center_x}, {center_y})...", flush=True)
        while valid_tiles_found < args.num_tiles and current_radius <= MAX_RADIUS:
            if valid_tiles_found == 0 and current_radius == 0:
                current_radius = 1 
            elif valid_tiles_found < args.num_tiles:
                current_radius += 1
            
            side_tiles = (current_radius * 2) + 1
            grid_px = tile_size + (side_tiles - 1) * stride
            sx = center_x - (grid_px // 2)
            sy = center_y - (grid_px // 2)
            
            temp_count = 0
            for r in range(side_tiles):
                for c in range(side_tiles):
                    gx = sx + (c * stride)
                    gy = sy + (r * stride)
                    try:
                        preview = slide.read_region((gx, gy), 2, (256, 256)).convert("RGB")
                        if is_valid_tile(preview):
                            temp_count += 1
                    except:
                        pass
            
            valid_tiles_found = temp_count
            print(f"  Radius {current_radius} ({side_tiles}x{side_tiles}): {valid_tiles_found} valid.", flush=True)
            if valid_tiles_found >= args.num_tiles:
                break
    else:
        grid_side = int(math.ceil(math.sqrt(args.num_tiles)))
        current_radius = grid_side // 2

    side_tiles = (current_radius * 2) + 1
    canvas_w = tile_size + (side_tiles - 1) * stride
    canvas_h = tile_size + (side_tiles - 1) * stride
    
    start_x = center_x - (canvas_w // 2)
    start_y = center_y - (canvas_h // 2)
    if start_x < 0:
        start_x = 0
    if start_y < 0:
        start_y = 0
    
    try:
        full_image = Image.new("RGB", (canvas_w, canvas_h), (240, 240, 240))
        full_mask = Image.new("L", (canvas_w, canvas_h), (0))
    except:
        print("Canvas memory error. Reducing...", flush=True)
        sys.exit(1)

    transform = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])
    saved_counter = 0
    
    print(f"Generating reconstruction...", flush=True)
    
    for r in range(side_tiles):
        for c in range(side_tiles):
            gx = start_x + (c * stride)
            gy = start_y + (r * stride)
            cx = c * stride
            cy = r * stride
            
            try:
                tile = slide.read_region((gx, gy), 0, (tile_size, tile_size)).convert("RGB")
            except:
                continue
            
            full_image.paste(tile, (cx, cy))
            
            valid = True
            if args.smart_filter and not is_valid_tile(tile):
                valid = False
            
            if args.demo_mode and r == side_tiles//2 and c == side_tiles//2:
                valid = True

            if valid:
                img_np = np.array(tile)
                inputs = transform(image=img_np)["image"].unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model(inputs).sigmoid()
                    mask_np = (pred > 0.5).float().cpu().numpy()[0, 0]
                
                mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).resize((tile_size, tile_size), Image.NEAREST)
                full_mask.paste(mask_pil, (cx, cy))
                
                if saved_counter < args.num_tiles:
                    green = Image.new("RGB", tile.size, (0, 255, 0))
                    ov = Image.composite(green, tile, mask_pil.convert("L"))
                    tile_vis = Image.blend(tile, ov, 0.3)
                    tile_vis.save(os.path.join(args.out, f"tile_{saved_counter}.jpg"), quality=80)
                    saved_counter += 1
                    print(f"  [Tile {saved_counter}] Saved.", flush=True)

    print("Saving final image...", flush=True)
    green_overlay = Image.new("RGB", full_image.size, (0, 255, 0))
    final_overlay = Image.composite(green_overlay, full_image, full_mask)
    final_reconstruction = Image.blend(full_image, final_overlay, 0.3)
    final_reconstruction.save(os.path.join(args.out, "full_reconstruction.jpg"), quality=85)
    
    print("Done.", flush=True)

if __name__ == "__main__":
    main()