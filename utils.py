

import os
import numpy as np

import glob
import torch
from torch.utils import data
import pandas as pd
from pathlib import Path
import params
from matplotlib import pyplot as plt
from matplotlib import gridspec


class ConditionalNpyDataset(data.Dataset):

    def __init__(
        self,
        npy_dir: str,
        csv_path: str,
        indices=None,
        y_mean: np.ndarray | None = None,
        y_std: np.ndarray | None = None,
    ):
        
        self.npy_dir = Path(npy_dir)
        self.files = list(self.npy_dir.glob("*.npy"))
        if len(self.files) == 0:
            raise RuntimeError(f"No .npy files found in {self.npy_dir}")

        df = pd.read_csv(csv_path, header=0)
        if len(df) != len(self.files):
            raise RuntimeError(
                f"CSV rows ({len(df)}) must match number of .npy files ({len(self.files)})."
            )

        self.Y_full = df.values.astype("float32")
        self.cond_names = list(df.columns)
        self.cond_dim = self.Y_full.shape[1]
        if indices is None:
            self.idx = np.arange(len(self.files), dtype=int)
        else:
            self.idx = np.array(indices, dtype=int)
       
        if (y_mean is None) or (y_std is None):
            y_mean = self.Y_full[self.idx].mean(axis=0)
            y_std = self.Y_full[self.idx].std(axis=0) + 1e-8

        self.y_mean = y_mean.astype("float32")
        self.y_std = y_std.astype("float32")

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        k = self.idx[i]
        vol = np.load(self.files[k]).astype("float32") 
        vol = torch.from_numpy(vol)                
        y = (self.Y_full[k] - self.y_mean) / self.y_std
        y = torch.from_numpy(y) 

        return vol, y


def generateZ(args, batch):
   
    return torch.randn(batch, params.z_dim, device=params.device)


def SavePloat_Voxels(voxels, path, iteration):
    voxels = voxels[:8] >= 0.7
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, f"{str(iteration).zfill(3)}.png"),
                bbox_inches='tight')
    plt.close()
