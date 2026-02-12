"""
Tile extraction and preprocessing module for whole slide images.

Implements regular grid-based tiling with padding and tissue content filtering.
"""

from PIL import Image
from background import compute_tissue_ratio


def pad_tile(tile, tile_size):
    """
    Pad tile to target size if smaller (preserves original at top-left).
    
    Args:
        tile (PIL.Image): Input tile image.
        tile_size (int): Target square tile dimension.
    
    Returns:
        tuple: (padded_tile, padded_flag) where padded_flag indicates if padding was applied.
    """
    w, h = tile.size
    padded = False

    if w < tile_size or h < tile_size:
        new_img = Image.new("RGB", (tile_size, tile_size))
        new_img.paste(tile, (0, 0))
        tile = new_img
        padded = True

    return tile, padded


def generate_tiles(slide, tile_size, tissue_threshold=0.05):
    """
    Generate tiles from slide with tissue ratio and informativeness filtering.
    
    Yields tile dictionaries in row-major order across full slide at full resolution.
    Automatically flags tiles as informative based on saturation-based tissue detection.
    
    Args:
        slide (openslide.OpenSlide): Opened whole slide image object.
        tile_size (int): Size of square tiles in pixels.
        tissue_threshold (float): Minimum tissue ratio to mark as informative. Default 0.05.
    
    Yields:
        dict: Tile metadata with keys 'x', 'y', 'padded', 'tissue_ratio',
              'is_informative', 'image' (PIL.Image).
    """
    width, height = slide.level_dimensions[0]

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):

            tile = slide.read_region((x, y), 0, (tile_size, tile_size))
            tile = tile.convert("RGB")

            tile, padded = pad_tile(tile, tile_size)
            tissue_ratio = compute_tissue_ratio(tile)
            is_informative = tissue_ratio > tissue_threshold

            yield {
                "x": x,
                "y": y,
                "padded": padded,
                "tissue_ratio": tissue_ratio,
                "is_informative": is_informative,
                "image": tile
            }
