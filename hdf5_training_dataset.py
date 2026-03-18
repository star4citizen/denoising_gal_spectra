from typing import Optional
import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class HDF5SpectrumDataset(Dataset):
    """
    HDF5에 저장된 정규화 clean/error 스펙트럼으로부터
    학습 시점에 S/N 기반 노이즈를 동적으로 추가합니다.
    """

    def __init__(
        self,
        hdf5_path: str,
        min_sn: int = 5,
        max_sn: int = 20,
        seed: int = 42,
    ) -> None:
        self.hdf5_path = hdf5_path
        self.min_sn = min_sn
        self.max_sn = max_sn
        self.seed = seed

        with h5py.File(self.hdf5_path, "r") as handle:
            self.length = int(handle["clean_flux"].shape[0])
            self.spectrum_length = int(handle["clean_flux"].shape[1])

        self._handle: Optional[h5py.File] = None
        self._clean_flux = None
        self._error = None
        self._wave = None

    def __len__(self) -> int:
        return self.length

    def _ensure_open(self) -> None:
        if self._handle is None:
            self._handle = h5py.File(self.hdf5_path, "r")
            self._wave = self._handle["wave"]
            self._clean_flux = self._handle["clean_flux"]
            self._error = self._handle["error"]

    def __getitem__(self, index: int):
        self._ensure_open()

        wave = np.asarray(self._wave[index], dtype=np.float32)
        clean_flux = np.asarray(self._clean_flux[index], dtype=np.float32)
        error = np.asarray(self._error[index], dtype=np.float32)

        sn_rng = random.Random(self.seed + index)
        sn_value = sn_rng.randint(self.min_sn, self.max_sn)

        noise_std = np.clip(error / float(sn_value), a_min=1e-9, a_max=None)
        noise_rng = np.random.default_rng(self.seed + index)
        noise = noise_rng.normal(loc=0.0, scale=noise_std).astype(np.float32)
        noisy_flux = clean_flux + noise

        return {
            "wave": torch.from_numpy(wave),
            "noisy_flux": torch.from_numpy(noisy_flux).unsqueeze(-1),
            "clean_flux": torch.from_numpy(clean_flux).unsqueeze(-1),
            "error": torch.from_numpy(error).unsqueeze(-1),
            "sn": torch.tensor(sn_value, dtype=torch.int64),
        }

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()


def create_dataloader(
    hdf5_path: str,
    batch_size: int = 64,
    min_sn: int = 5,
    max_sn: int = 20,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    dataset = HDF5SpectrumDataset(
        hdf5_path=hdf5_path,
        min_sn=min_sn,
        max_sn=max_sn,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    loader = create_dataloader(
        hdf5_path="/workspace/git/train_90000_random.h5",
        batch_size=8,
        min_sn=5,
        max_sn=20,
        shuffle=True,
    )
    first_batch = next(iter(loader))
    print("Batch keys:", list(first_batch.keys()))
    for key, value in first_batch.items():
        if hasattr(value, "shape"):
            print(f"{key}: {tuple(value.shape)}")
        else:
            print(f"{key}: {value}")
