import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class KidneyDataset(Dataset):

    def __init__(self, root_dir, img_size=256, augment=False, clahe=False):

        self.img_dir = os.path.join(root_dir, "image")
        self.mask_dir = os.path.join(root_dir, "mask")

        self.images = sorted(os.listdir(self.img_dir))
        self.img_size = img_size
        self.augment = augment
        self.clahe = clahe

    def __len__(self):
        return len(self.images)

    def _apply_clahe(self, img):

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        return clahe.apply(img)

    def _apply_augmentation(self, img, mask):

        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        if np.random.rand() < 0.3:
            angle = np.random.uniform(-12.0, 12.0)
            center = (self.img_size / 2, self.img_size / 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            img = cv2.warpAffine(
                img,
                matrix,
                (self.img_size, self.img_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )
            mask = cv2.warpAffine(
                mask,
                matrix,
                (self.img_size, self.img_size),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        if np.random.rand() < 0.3:
            alpha = np.random.uniform(0.9, 1.15)
            beta = np.random.uniform(-0.06, 0.06)
            img = np.clip(img * alpha + beta, 0.0, 1.0)

        if np.random.rand() < 0.2:
            noise = np.random.normal(0.0, 0.015, size=img.shape)
            img = np.clip(img + noise, 0.0, 1.0)

        return img, mask

    def __getitem__(self, idx):

        img_name = self.images[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        if self.clahe:
            img = self._apply_clahe(img)

        img = img.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        if self.augment:
            img, mask = self._apply_augmentation(img, mask)

        img = np.stack([img, img, img], axis=0)

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32)
        )

