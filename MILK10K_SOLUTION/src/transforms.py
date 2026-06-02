"""
transforms.py — Albumentations-based augmentation pipelines for MILK10k.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Heavy augmentation for training."""
    return A.Compose([
        A.RandomResizedCrop(
            height=image_size,
            width=image_size,
            scale=(0.7, 1.0),
            ratio=(0.75, 1.333),
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0,
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=1.0,
            ),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            p=0.4,
        ),
        A.CoarseDropout(
            max_holes=8,
            max_height=image_size // 8,
            max_width=image_size // 8,
            min_holes=1,
            fill_value=0,
            p=0.3,
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Minimal deterministic transform for validation and test."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


def get_tta_transforms(image_size: int = 224) -> list[A.Compose]:
    """
    Test-Time Augmentation (TTA) variants.
    Returns a list of transforms; run inference with each and average probabilities.
    """
    base = A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    resize = A.Resize(height=image_size, width=image_size)

    variants = [
        # Original
        A.Compose([resize, base, ToTensorV2()]),
        # Horizontal flip
        A.Compose([resize, A.HorizontalFlip(p=1.0), base, ToTensorV2()]),
        # Vertical flip
        A.Compose([resize, A.VerticalFlip(p=1.0), base, ToTensorV2()]),
        # 90° rotation
        A.Compose([resize, A.Rotate(limit=(90, 90), p=1.0), base, ToTensorV2()]),
    ]
    return variants
