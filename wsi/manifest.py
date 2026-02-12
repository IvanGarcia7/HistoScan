"""
Manifest generation module for tile metadata storage.

Exports tile extraction metadata to CSV format for downstream analysis and tracking.
"""

import pandas as pd


def save_manifest(tiles, output_csv):
    """
    Save tile metadata to CSV manifest file.
    
    Converts list of tile dictionaries to DataFrame and exports location,
    padding, tissue ratio, and informativeness flags.
    
    Args:
        tiles (list): List of dictionaries with keys 'x', 'y', 'padded',
                     'tissue_ratio', 'is_informative'.
        output_csv (str): Path where CSV file will be saved.
    """
    rows = []

    for t in tiles:
        rows.append({
            "x": t["x"],
            "y": t["y"],
            "padded": t["padded"],
            "tissue_ratio": t["tissue_ratio"],
            "is_informative": t["is_informative"]
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
