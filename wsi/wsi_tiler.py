"""
Whole slide image tiling and metadata extraction pipeline.

Orchestrates tile generation from WSI files with tissue filtering,
metadata collection, and pyramid structure inspection.
"""

import argparse
import os
import openslide
from tqdm import tqdm
from PIL import Image

from pyramid import inspect_pyramid
from tiler import generate_tiles
from manifest import save_manifest
from logging_config import configure_logging


def main():
    """
    Main entry point for WSI tiling pipeline.
    
    Orchestrates:
    1. Pyramid structure inspection and export
    2. Regular grid-based tile extraction with tissue filtering
    3. Selective tile saving based on informativeness and filters
    4. Metadata CSV generation for downstream processing
    
    Supports demo mode (limited tiles) and empty tile filtering.
    Cleans image objects from memory after saving to minimize RAM usage.
    """
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Extract tiles from whole slide images with tissue filtering."
    )
    parser.add_argument("--slide", required=True, help="Path to WSI file.")
    parser.add_argument("--out", required=True, help="Output directory for tiles and metadata.")
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size in pixels (default: 1024).")
    parser.add_argument("--demo", action="store_true", help="Demo mode: limit to 200 tiles.")
    parser.add_argument("--max-tiles", type=int, default=None, help="Maximum tiles to extract (overrides demo).")
    parser.add_argument("--filter-empty", action="store_true", help="Skip tiles below tissue threshold.")
    
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sample_dir = os.path.join(args.out, "tiles_sample")
    os.makedirs(sample_dir, exist_ok=True)

    inspect_pyramid(args.slide, os.path.join(args.out, "pyramid.json"))

    slide = openslide.OpenSlide(args.slide)

    if args.max_tiles is not None:
        max_iter = args.max_tiles
    elif args.demo:
        max_iter = 200
    else:
        max_iter = None 

    print(f"--- Processing started. Filter: {args.filter_empty}. Target: {max_iter if max_iter else 'ALL'} ---")

    tiles_metadata = [] 
    generator = generate_tiles(slide, args.tile_size)
    count_saved = 0

    for i, tile_data in enumerate(tqdm(generator)):
        
        is_informative = tile_data.get("is_informative", True)

        if args.filter_empty and not is_informative:
            if "image" in tile_data:
                del tile_data["image"]
            continue 

        image_obj = tile_data.get("image")

        if image_obj:
            filename = f"tile_{i:04d}_x{tile_data['x']}_y{tile_data['y']}.jpg"
            save_path = os.path.join(sample_dir, filename)
            
            try:
                if image_obj.mode != "RGB":
                    image_obj = image_obj.convert("RGB")
                
                image_obj.save(save_path, "JPEG", quality=90)
                
                tile_data["filename"] = filename
                del tile_data["image"]
                
                tiles_metadata.append(tile_data)
                count_saved += 1
                
            except Exception as e:
                print(f"Error saving tile {filename}: {e}")

        if max_iter and count_saved >= max_iter:
            print(f"Target of {max_iter} tiles reached.")
            break

    save_manifest(tiles_metadata, os.path.join(args.out, "manifest.csv"))
    
    slide.close()
    print(f"--- Processing complete. {count_saved} tiles generated. ---")


if __name__ == "__main__":
    main()