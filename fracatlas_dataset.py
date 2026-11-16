import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FracAtlasFractureDataset(Dataset):
    """
    Binary Fracture Segmentation Dataset for FracAtlas:
      Class 0: Background
      Class 1: Fracture Region
    """
    def __init__(
        self,
        split="train",
        root_dir=".",
        img_size=(256, 256),
        apply_clahe=True,
        augment=True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.apply_clahe = apply_clahe
        self.augment = augment and (split == "train")
        self.split = split

        # Official fracture splits from FracAtlas
        split_csv_path = os.path.join(root_dir, "Utilities", "Fracture Split", f"{split}.csv")
        if not os.path.exists(split_csv_path):
            raise FileNotFoundError(f"Split CSV not found at: {split_csv_path}")

        split_df = pd.read_csv(split_csv_path)
        self.image_ids = split_df["image_id"].tolist()

        self.images_dir = os.path.join(root_dir, "images", "Fractured")
        self.masks_dir = os.path.join(root_dir, "fracture_masks")

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if apply_clahe else None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, img_name)

        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_name)

        # 1. Load Image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at: {img_path}")

        # 2. Load Binary Fracture Mask
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = np.zeros(image.shape, dtype=np.uint8)

        # 3. CLAHE Contrast Enhancement
        if self.clahe is not None:
            image = self.clahe.apply(image)

        # 4. Standardized Resize
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # 5. Data Augmentation (Train split only)
        if self.augment:
            if np.random.rand() > 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)
            if np.random.rand() > 0.5:
                image = np.flipud(image)
                mask = np.flipud(mask)
            if np.random.rand() > 0.5:
                k = np.random.choice([1, 2, 3])
                image = np.rot90(image, k)
                mask = np.rot90(mask, k)

        # 6. Normalize to [0.0, 1.0]
        image = (image.astype(np.float32) / 255.0)
        mask = mask.astype(np.float32)

        # Convert to PyTorch Tensors
        # Image: [1, H, W]
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).float()
        # Mask: [1, H, W]
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).float()

        return image_tensor, mask_tensor
