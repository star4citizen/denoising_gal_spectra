import math
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

from hdf5_training_dataset import HDF5SpectrumDataset


class HDF5SpectrumDataModule(pl.LightningDataModule):
    """
    /workspace/git/train_90000_random.h5 같은 단일 HDF5 파일을
    PyTorch Lightning 학습에 바로 연결하기 위한 DataModule.
    """

    def __init__(
        self,
        hdf5_path: str,
        batch_size: int = 64,
        train_fraction: float = 0.9,
        min_sn: int = 5,
        max_sn: int = 20,
        val_min_sn: Optional[int] = None,
        val_max_sn: Optional[int] = None,
        num_workers: int = 0,
        seed: int = 42,
        pin_memory: Optional[bool] = None,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")

        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.train_fraction = train_fraction
        self.min_sn = min_sn
        self.max_sn = max_sn
        self.val_min_sn = min_sn if val_min_sn is None else val_min_sn
        self.val_max_sn = max_sn if val_max_sn is None else val_max_sn
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0

        self.train_dataset = None
        self.val_dataset = None
        self.train_indices = None
        self.val_indices = None
        self.dataset_size = None

    def prepare_data(self) -> None:
        dataset = HDF5SpectrumDataset(
            hdf5_path=self.hdf5_path,
            min_sn=self.min_sn,
            max_sn=self.max_sn,
            seed=self.seed,
        )
        self.dataset_size = len(dataset)
        dataset.close()

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset_size is None:
            self.prepare_data()

        if self.train_indices is None or self.val_indices is None:
            generator = torch.Generator().manual_seed(self.seed)
            permutation = torch.randperm(self.dataset_size, generator=generator).tolist()
            train_size = int(math.floor(self.dataset_size * self.train_fraction))
            self.train_indices = permutation[:train_size]
            self.val_indices = permutation[train_size:]

        if stage in (None, "fit"):
            train_base = HDF5SpectrumDataset(
                hdf5_path=self.hdf5_path,
                min_sn=self.min_sn,
                max_sn=self.max_sn,
                seed=self.seed,
            )
            val_base = HDF5SpectrumDataset(
                hdf5_path=self.hdf5_path,
                min_sn=self.val_min_sn,
                max_sn=self.val_max_sn,
                seed=self.seed + 10_000_000,
            )
            self.train_dataset = Subset(train_base, self.train_indices)
            self.val_dataset = Subset(val_base, self.val_indices)

        if stage == "validate" and self.val_dataset is None:
            val_base = HDF5SpectrumDataset(
                hdf5_path=self.hdf5_path,
                min_sn=self.val_min_sn,
                max_sn=self.val_max_sn,
                seed=self.seed + 10_000_000,
            )
            self.val_dataset = Subset(val_base, self.val_indices)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized. Call setup('fit') first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized. Call setup('fit') or setup('validate') first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        for subset in (self.train_dataset, self.val_dataset):
            if subset is not None and hasattr(subset, "dataset") and hasattr(subset.dataset, "close"):
                subset.dataset.close()


if __name__ == "__main__":
    dm = HDF5SpectrumDataModule(
        hdf5_path="/workspace/git/train_90000_random.h5",
        batch_size=8,
        train_fraction=0.9,
        min_sn=5,
        max_sn=20,
        num_workers=0,
        seed=42,
    )
    dm.prepare_data()
    dm.setup("fit")
    train_batch = next(iter(dm.train_dataloader()))
    val_batch = next(iter(dm.val_dataloader()))
    print("dataset_size:", dm.dataset_size)
    print("train_size:", len(dm.train_indices))
    print("val_size:", len(dm.val_indices))
    print("train batch:", {k: tuple(v.shape) for k, v in train_batch.items()})
    print("val batch:", {k: tuple(v.shape) for k, v in val_batch.items()})
