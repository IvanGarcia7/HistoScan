"""
Tile sampling module for balanced WSI dataset creation.

Filters and randomly samples informative tiles from whole slide images
for downstream processing and analysis.
"""

import random
import os


def save_sampled_tiles(tiles, output_dir, max_tiles=300, seed=42):
    """
    Sample and save informative tiles to disk.
    
    Filters tiles marked as informative, randomly shuffles them with fixed seed
    for reproducibility, and saves up to max_tiles PNG images.
    
    Args:
        tiles (list): List of tile dictionaries with 'is_informative', 'image',
                     'x', 'y' keys.
        output_dir (str): Directory where tile images will be saved.
        max_tiles (int): Maximum number of tiles to sample. Default is 300.
        seed (int): Random seed for reproducible shuffling. Default is 42.
    
    Returns:
        list: List of selected tile dictionaries (subset of input).
    """
    os.makedirs(output_dir, exist_ok=True)

    random.seed(seed)

    informative = [t for t in tiles if t["is_informative"]]
    random.shuffle(informative)

    selected = informative[:max_tiles]

    for i, tile in enumerate(selected):
        filename = f"tile_{i}_x{tile['x']}_y{tile['y']}.png"
        tile["image"].save(os.path.join(output_dir, filename))

    return selected
