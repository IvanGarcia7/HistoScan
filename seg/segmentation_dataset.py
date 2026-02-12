"""
Image segmentation dataset module.

Provides PyTorch Dataset implementations for semantic segmentation tasks.
Supports:
- Loading image and mask pairs from separate directories
- Multiple preprocessing pipelines (ImageNet normalization, Albumentations)
- Inference mode (images only) and training mode (images + masks)
- Flexible transform configuration with automatic tensor conversion
"""

from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for semantic segmentation with flexible preprocessing.
    
    Loads image-mask pairs from separate directories and applies configurable
    transformations. Supports three preprocessing modes: ImageNet normalization,
    Albumentations augmentation, and raw tensor conversion. Handles both training
    (with masks) and inference (images only) scenarios.
    
    Attributes:
        images (list): Sorted list of image file paths.
        masks (list or None): Sorted list of mask file paths, None if inference mode.
        transform (callable, str, or None): Preprocessing pipeline. Can be:
            - "imagenet": ImageNet normalization (mean/std)
            - Albumentations Compose: Augmentation + tensor conversion
            - None: Basic tensor conversion without augmentation
    """
    def __init__(self, images_dir, masks_dir=None, transform=None):
        """
        Initialize the segmentation dataset.
        
        Args:
            images_dir (str or Path): Directory containing image files.
            masks_dir (str, Path, or None): Directory containing mask files.
                If None, dataset operates in inference mode (images only).
                Default is None.
            transform (callable, str, or None): Preprocessing pipeline.
                - "imagenet": Apply ImageNet normalization (CIFAR-10 statistics)
                - Albumentations Compose object: Apply augmentations and convert to tensor
                - None: Convert to tensor without augmentation (default)
        
        Note:
            Images and masks are automatically sorted to ensure consistent pairing.
            Filename order must match between images_dir and masks_dir for correct alignment.
        """
        self.images = sorted(Path(images_dir).glob("*"))
        self.masks = sorted(Path(masks_dir).glob("*")) if masks_dir else None
        self.transform = transform

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Load and preprocess a single sample from the dataset.
        
        Reads image (and optionally mask) at the given index, applies preprocessing
        according to the configured transform mode, and ensures proper tensor formats.
        Automatically handles normalization and shape alignment for both training
        and inference modes.
        
        Args:
            idx (int): Index of the sample to retrieve (0-based).
        
        Returns:
            If masks are available (training mode):
                tuple: (image_tensor, mask_tensor) where
                    - image_tensor (torch.Tensor): Shape [3, H, W], dtype float32, normalized
                    - mask_tensor (torch.Tensor): Shape [1, H, W], dtype float32, values in [0, 1]
            If no masks (inference mode):
                torch.Tensor: Image tensor, Shape [3, H, W], dtype float32
        
        Raises:
            FileNotFoundError: If image or mask file cannot be opened.
            IndexError: If idx is out of dataset bounds.
        """
        img = np.array(Image.open(self.images[idx]).convert("RGB"))

        if self.masks:
            mask = np.array(Image.open(self.masks[idx]).convert("L"))

            if mask.ndim == 2:
                mask = mask[..., np.newaxis]

            if self.transform == "imagenet":
                img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = (img - mean) / std

                mask = torch.tensor(mask).permute(2, 0, 1).float() / 255.0

            elif self.transform:
                augmented = self.transform(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]

                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask.astype(np.float32))

                if mask.max() > 1.0:
                    mask = mask / 255.0

                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                elif mask.ndim == 3 and mask.shape[2] == 1:
                    mask = mask.permute(2, 0, 1)

            else:
                img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
                mask = torch.tensor(mask).permute(2, 0, 1).float() / 255.0

            mask = mask.float()
            img = img.float()

            return img, mask

        else:
            if self.transform == "imagenet":
                img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = (img - mean) / std
            elif self.transform:
                img = self.transform(image=img)["image"]
            else:
                img = torch.tensor(img).permute(2, 0, 1).float() / 255.0

            img = img.float()
            return img
