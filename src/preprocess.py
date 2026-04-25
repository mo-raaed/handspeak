"""
preprocess.py
=============
Data loading, augmentation, and splitting pipeline for ASL gesture recognition.

Loads images from the ASL Alphabet dataset using torchvision.datasets.ImageFolder,
applies class-appropriate transforms, splits into train/validation sets (85/15),
and returns ready-to-use DataLoaders.

Usage:
    # Standalone test
    python src/preprocess.py

    # Import from train.py
    from preprocess import get_dataloaders
    train_loader, val_loader, class_names = get_dataloaders()
"""

import os
import pickle

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path is relative to the project root (one level up from src/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, os.pardir)

# Dataset location — configurable. Change this if your data lives elsewhere.
DATASET_PATH = os.path.normpath(
    os.path.join(_PROJECT_ROOT, "data", "asl_alphabet_train", "asl_alphabet_train")
)

# Where to persist train/val split indices for reproducibility
SPLITS_PATH = os.path.normpath(
    os.path.join(_PROJECT_ROOT, "data", "splits.pkl")
)

# Hyperparameters
IMAGE_SIZE = 64
BATCH_SIZE = 32
NUM_WORKERS = 4
TRAIN_RATIO = 0.85          # 85% train, 15% validation
RANDOM_SEED = 42

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

# Training transform — includes data augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Validation transform — deterministic, no augmentation
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_splits(train_indices, val_indices, path):
    """Persist split indices to disk so every run uses the same split."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"train": train_indices, "val": val_indices}, f)


def _load_splits(path):
    """Load previously saved split indices."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["train"], data["val"]


class _TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset so we can apply a *different* transform per split.

    ImageFolder only accepts one transform at init time, but we need
    augmentation on training data and plain resize on validation data.
    This thin wrapper overrides the transform at read time.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        # `image` comes back as a PIL Image because the base dataset
        # was loaded with transform=None.
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def get_dataloaders(
    dataset_path=DATASET_PATH,
    splits_path=SPLITS_PATH,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    seed=RANDOM_SEED,
):
    """Load the ASL Alphabet dataset and return train/val DataLoaders.

    Parameters
    ----------
    dataset_path : str
        Root directory that contains one sub-folder per class (A/ … Z/).
    splits_path : str
        File path to save/load the train/val split indices.
    batch_size : int
        Mini-batch size for both loaders.
    num_workers : int
        Number of parallel data-loading workers.
    seed : int
        Random seed for the train/val split (ignored if splits file exists).

    Returns
    -------
    train_loader : DataLoader
    val_loader   : DataLoader
    class_names  : list[str]   — e.g. ['A', 'B', ..., 'Z']
    """

    # ------------------------------------------------------------------
    # 1. Load the full dataset WITHOUT transforms (applied per-split later)
    # ------------------------------------------------------------------
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=None)
    class_names = full_dataset.classes          # ['A', 'B', ..., 'Z']
    total_images = len(full_dataset)

    # ------------------------------------------------------------------
    # 2. Compute or load train/val split indices
    # ------------------------------------------------------------------
    if os.path.exists(splits_path):
        print(f"[preprocess] Loading existing split from {splits_path}")
        train_indices, val_indices = _load_splits(splits_path)
    else:
        print(f"[preprocess] Creating new 85/15 split (seed={seed})")
        train_size = int(total_images * TRAIN_RATIO)
        val_size = total_images - train_size

        # random_split returns Subset objects; we extract their indices
        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
        train_indices = train_subset.indices
        val_indices = val_subset.indices

        # Persist to disk for reproducibility
        _save_splits(train_indices, val_indices, splits_path)
        print(f"[preprocess] Split indices saved to {splits_path}")

    # ------------------------------------------------------------------
    # 3. Build Subsets with per-split transforms
    # ------------------------------------------------------------------
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    train_dataset = _TransformSubset(train_subset, transform=train_transform)
    val_dataset = _TransformSubset(val_subset, transform=val_transform)

    # ------------------------------------------------------------------
    # 4. Create DataLoaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,               # shuffle training data every epoch
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,               # no need to shuffle validation
        num_workers=num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 5. Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("  ASL Alphabet Dataset — Preprocessing Summary")
    print("=" * 50)
    print(f"  Dataset path  : {dataset_path}")
    print(f"  Image size    : {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"  Total images  : {total_images:,}")
    print(f"  Train split   : {len(train_indices):,}  ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Val split     : {len(val_indices):,}  ({(1-TRAIN_RATIO)*100:.0f}%)")
    print(f"  Classes ({len(class_names)}): {', '.join(class_names)}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Train batches : {len(train_loader):,}")
    print(f"  Val batches   : {len(val_loader):,}")
    print("=" * 50 + "\n")

    return train_loader, val_loader, class_names


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_loader, val_loader, class_names = get_dataloaders()

    # Grab one batch to confirm everything works end-to-end
    images, labels = next(iter(train_loader))
    print(f"[test] Sample batch — images: {images.shape}, labels: {labels.shape}")
    print(f"[test] Label examples: {labels[:8].tolist()}")
    print(f"[test] Pixel range  : [{images.min():.3f}, {images.max():.3f}]")
    print("[test] ✓ DataLoaders are ready.")
