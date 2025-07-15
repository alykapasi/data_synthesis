import numpy as np
import torch
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    """
    Dataset for preprocessed ECG beats saved in a .npy file.
    Each beat is expected to be a 1D array of fixed length (e.g., 250).
    """
    def __init__(self, npy_path: str):
        self.data = np.load(npy_path).astype(np.float32)
        self.data = torch.from_numpy(self.data).unsqueeze(1)  # shape: [N, 1, T]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # shape: [1, T]
