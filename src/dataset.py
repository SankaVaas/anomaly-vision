"""
MVTec AD dataset loader.
Handles download (via Kaggle or direct), train/test splits,
and returns DataLoaders ready for feature extraction.
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]


class MVTecDataset(Dataset):
    """
    Loads MVTec AD for a single category.
    train=True  → only normal (good) images for memory bank construction
    train=False → all test images + ground-truth masks for evaluation
    """

    def __init__(self, root: str, category: str, image_size: int = 256, train: bool = True):
        assert category in MVTEC_CATEGORIES, f"Unknown category: {category}"
        self.train = train
        self.image_size = image_size

        base = Path(root) / category
        if train:
            self.image_paths = sorted((base / "train" / "good").glob("*.png"))
            self.mask_paths = [None] * len(self.image_paths)
            self.labels = [0] * len(self.image_paths)
        else:
            self.image_paths, self.mask_paths, self.labels = [], [], []
            test_dir = base / "test"
            for defect_type in sorted(test_dir.iterdir()):
                label = 0 if defect_type.name == "good" else 1
                for img_path in sorted(defect_type.glob("*.png")):
                    self.image_paths.append(img_path)
                    self.labels.append(label)
                    if label == 1:
                        mask_path = base / "ground_truth" / defect_type.name / (img_path.stem + "_mask.png")
                        self.mask_paths.append(mask_path if mask_path.exists() else None)
                    else:
                        self.mask_paths.append(None)

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.img_transform(image)

        mask_path = self.mask_paths[idx]
        if mask_path and Path(mask_path).exists():
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)

        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "path": str(self.image_paths[idx]),
        }


def get_dataloaders(cfg: dict):
    """Build train and test DataLoaders from config dict."""
    data_cfg = cfg["data"]
    train_ds = MVTecDataset(
        root=data_cfg["root"],
        category=data_cfg["category"],
        image_size=data_cfg["image_size"],
        train=True,
    )
    test_ds = MVTecDataset(
        root=data_cfg["root"],
        category=data_cfg["category"],
        image_size=data_cfg["image_size"],
        train=False,
    )
    train_loader = DataLoader(train_ds, batch_size=data_cfg["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1,
                             shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader