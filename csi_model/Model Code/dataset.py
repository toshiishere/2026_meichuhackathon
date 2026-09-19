# ESP-Fi HAR Dataset Loader
# Input size: 1 × 950 × 52
# Modality: CSI Amplitude (CSIamp)

import os
import glob
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader


class ESP_Fi_HAR_Dataset(Dataset):
    """
    ESP-Fi HAR Dataset

    Directory structure:
    root_dir/
        train_amp/
            arm_wave/
            fall/
            jump/
            run/
            squat/
            turn/
            walk/
        test_amp/
            arm_wave/
            fall/
            jump/
            run/
            squat/
            turn/
            walk/


    Each .mat file should contain:
        CSIamp: ndarray with shape (950, 52) or (52, 950)

    Output:
        x: Tensor of shape (1, 950, 52)
        y: LongTensor label
    """

    def __init__(self,
                 root_dir: str,
                 split: str = "train_amp",
                 modal: str = "CSIamp",
                 transform=None):
        """
        Args:
            root_dir (str): Root directory of ESP-Fi HAR dataset
            split (str): 'train_amp' or 'test_amp'
            modal (str): Data modality key in .mat file (default: CSIamp)
            transform (callable, optional): Optional transform applied to input
        """
        self.root_dir = root_dir
        self.split = split
        self.modal = modal
        self.transform = transform

        self.activities = [
            'arm_wave',
            'fall',
            'jump',
            'run',
            'squat',
            'turn',
            'walk'
        ]


        self.data = []
        self.labels = []

        self._check_dataset_structure()
        self._load_dataset()

    def _check_dataset_structure(self):
        """Check whether dataset directory structure is valid"""
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")

        split_dir = os.path.join(self.root_dir, self.split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for act in self.activities:
            act_path = os.path.join(split_dir, act)
            if not os.path.exists(act_path):
                raise FileNotFoundError(f"Activity folder missing: {act_path}")

    def _load_dataset(self):
        """Load all CSI samples into memory"""
        print(f"[ESP-Fi HAR] Loading {self.split} set...")
        print("[ESP-Fi HAR] Activity order:", self.activities)
        
        split_dir = os.path.join(self.root_dir, self.split)

        for label, act in enumerate(self.activities):
            act_dir = os.path.join(split_dir, act)
            mat_files = glob.glob(os.path.join(act_dir, "*.mat"))

            for mat_path in mat_files:
                try:
                    mat = sio.loadmat(mat_path)

                    if self.modal not in mat:
                        print(f"Warning: '{self.modal}' not found in {mat_path}")
                        continue

                    x = mat[self.modal]

                    # Ensure shape = (T, 52) for whatever window length T this
                    # sample was built with -- not hardcoded to the paper's
                    # 950 (10s @ ~100Hz); a shorter/longer window just changes T.
                    if x.shape[1] == 52:
                        pass
                    elif x.shape[0] == 52:
                        x = x.T
                    else:
                        raise ValueError(
                            f"Unexpected shape {x.shape} in {mat_path} "
                            f"(neither dimension is 52 subcarriers)"
                        )
                    t = x.shape[0]

                    # Z-score normalization
                    x = (x - np.mean(x)) / (np.std(x) + 1e-8)

                    # Reshape to (1, T, 52)
                    x = x.reshape(1, t, 52).astype(np.float32)

                    self.data.append(x)
                    self.labels.append(label)

                except Exception as e:
                    print(f"Error loading {mat_path}: {e}")

        self.data = np.asarray(self.data, dtype=np.float32)
        self.labels = np.asarray(self.labels, dtype=np.int64)

        if len(self.data) == 0:
            raise RuntimeError("No valid data samples loaded.")

        print(f"[ESP-Fi HAR] Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            x = self.transform(x)

        return x, y


def get_dataloader(root_dir,
                   split="test_amp",
                   batch_size=64,
                   shuffle=False,
                   num_workers=0):
    """
    Build DataLoader for ESP-Fi HAR

    Args:
        root_dir (str): Dataset root directory
        split (str): 'train_amp' or 'test_amp'
        batch_size (int): Batch size
        shuffle (bool): Shuffle data
        num_workers (int): DataLoader workers

    Returns:
        torch.utils.data.DataLoader
    """
    dataset = ESP_Fi_HAR_Dataset(
        root_dir=root_dir,
        split=split,
        modal="CSIamp"
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )



