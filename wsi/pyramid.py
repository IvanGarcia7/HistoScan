"""
Pyramid structure inspection module for whole slide images.

Extracts multi-resolution level metadata from WSI files and exports
pyramid hierarchy information for analysis.
"""

import openslide
import json


def inspect_pyramid(slide_path, output_json):
    """
    Inspect and export WSI pyramid structure to JSON.
    
    Extracts dimension and downsampling information for each resolution level
    in the multi-scale image pyramid typical of whole slide formats.
    
    Args:
        slide_path (str): Path to WSI file.
        output_json (str): Path where pyramid metadata JSON will be saved.
    
    Returns:
        dict: Pyramid structure with 'level_count' and 'levels' array containing
              level index, width, height, and downsample factor.
    """
    slide = openslide.OpenSlide(slide_path)

    data = {
        "level_count": slide.level_count,
        "levels": []
    }

    for i in range(slide.level_count):
        dims = slide.level_dimensions[i]
        downsample = float(slide.level_downsamples[i])

        data["levels"].append({
            "level": i,
            "width": dims[0],
            "height": dims[1],
            "downsample": round(downsample, 2)
        })

    with open(output_json, "w") as f:
        json.dump(data, f, indent=4)

    slide.close()
    return data