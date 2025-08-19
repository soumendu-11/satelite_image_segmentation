import os
import cv2
import numpy as np
import pandas as pd
from patchify import patchify
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset

# Parameters
patch_size = 256
scaler = MinMaxScaler()

# Color to Class Mapping
def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return np.array(tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4)))

COLOR_MAP = {
    0: hex_to_rgb('#3C1098'),  # Building
    1: hex_to_rgb('#8429F6'),  # Land
    2: hex_to_rgb('#6EC1E4'),  # Road
    3: hex_to_rgb('#FEDD3A'),  # Vegetation
    4: hex_to_rgb('#E2A929'),  # Water
    5: hex_to_rgb('#9B9B9B')   # Unlabeled
}

def rgb_to_2D_label(label):
    """Convert RGB mask to 2D class indices."""
    label_seg = np.zeros(label.shape[:2], dtype=np.uint8)
    for k, v in COLOR_MAP.items():
        matches = np.all(label == v, axis=-1)
        label_seg[matches] = k
    return label_seg


class SegmentationDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.image_patches = []
        self.mask_patches = []
        self._prepare_data()

    def _prepare_data(self):
        for _, row in self.df.iterrows():
            img_path = row['Image']
            mask_path = row['Mask']

            # Load and crop image
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            img = Image.fromarray(img)
            img = img.crop((0, 0, (w // patch_size) * patch_size, (h // patch_size) * patch_size))
            img = np.array(img)
            img_patches = patchify(img, (patch_size, patch_size, 3), step=patch_size)

            # Load and crop mask
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = Image.fromarray(mask)
            mask = mask.crop((0, 0, (w // patch_size) * patch_size, (h // patch_size) * patch_size))
            mask = np.array(mask)
            mask_patches = patchify(mask, (patch_size, patch_size, 3), step=patch_size)

            for i in range(img_patches.shape[0]):
                for j in range(img_patches.shape[1]):
                    img_patch = img_patches[i, j, 0]
                    img_patch = scaler.fit_transform(img_patch.reshape(-1, 3)).reshape(img_patch.shape)
                    mask_patch = rgb_to_2D_label(mask_patches[i, j, 0])
                    self.image_patches.append(img_patch)
                    self.mask_patches.append(mask_patch)

    def __len__(self):
        return len(self.image_patches)

    def __getitem__(self, idx):
        image = torch.tensor(self.image_patches[idx], dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(self.mask_patches[idx], dtype=torch.long)
        return image, mask

