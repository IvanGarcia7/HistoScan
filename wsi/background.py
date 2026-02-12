"""
Background and tissue detection module for whole slide image analysis.

Quantifies tissue content in microscopy images using HSV color space saturation.
"""

import numpy as np
import cv2


def compute_tissue_ratio(pil_image):
    """
    Calculate the proportion of tissue pixels in an image using saturation analysis.
    
    Tissue regions have higher saturation in HSV due to staining, allowing
    discrimination from background/white areas. Uses empirically tuned threshold
    (saturation > 20) suitable for standard histopathology stains (H&E, etc).
    
    Args:
        pil_image (PIL.Image): Input image in RGB format.
    
    Returns:
        float: Tissue ratio in [0.0, 1.0]. Example: 0.15 = 15% tissue coverage.
    """
    img = np.array(pil_image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    saturation = hsv[:, :, 1]
    tissue_mask = saturation > 20

    return float(tissue_mask.sum() / tissue_mask.size)
