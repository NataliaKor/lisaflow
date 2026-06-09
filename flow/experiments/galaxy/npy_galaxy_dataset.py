import numpy as np
import torch
from torch.utils.data import Dataset

class NPYGalaxyDataset(Dataset):
    """PyTorch Dataset for NPY files with columns:
    [Amplitude, Frequency, FrequencyDerivative, EquatorialLongitude, EquatorialLatitude]

    By default this loads the whole file into memory. For very large files, pass
    ``mmap_mode='r'`` to ``np.load`` to memory-map instead.
    """

    def __init__(self, npy_path, mmap_mode=None, dtype=np.float64):
        self.npy_path = npy_path
        self._samples = np.load(npy_path, mmap_mode=mmap_mode).astype(dtype)
        if self._samples.ndim != 2 or self._samples.shape[1] != 5:
            raise ValueError(f"Expected NPY array with shape (N,5), got {self._samples.shape}")

        self._samples[:, 0] = np.log(self._samples[:, 0]) # Ampliture
        self._samples[:, 1] = np.log(self._samples[:, 1]) # Frequency
        self._samples[:, 2] =  -np.sign(self._samples[:, 2])*np.log(np.abs(self._samples[:, 2])) # Frequency derivative 
        self._samples[:, -1] = np.sin(self._samples[:, -1]) # EquatorialLatitude
        self._min = self._samples.min(axis=0)
        self._max = self._samples.max(axis=0)
        print('self._min.shape = ', self._min.shape)

    def __len__(self):
        return self._samples.shape[0]

    def __getitem__(self, idx):
        
        denom = (self._max - self._min)
        output = torch.from_numpy(2.0 * (self._samples[idx] - self._min) / denom - 1.0)
        return output

    @property
    def samples_min(self):
        return getattr(self, "_min", None)

    @property
    def samples_max(self):
        return getattr(self, "_max", None)

    @property
    def labels(self):
        return ["A", "f0", "fdot", "beta", "lam"]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick test for NpyGalaxyDataset")
    parser.add_argument("npyfile", help="Path to the NPY file to load")
    args = parser.parse_args()

    ds = NPYGalaxyDataset(args.npyfile, normalize=True)
    print("Loaded:", args.npyfile)
    print("Number of samples:", len(ds))
    print("Sample[0]:", ds[0])

