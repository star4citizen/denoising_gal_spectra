import argparse
import os
import random
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch


def normalize_spectrum(clean_flux: np.ndarray, error: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    norm_range_end = min(1300, len(clean_flux))
    norm_range_start = min(1200, norm_range_end - 1)
    if norm_range_end > norm_range_start:
        norm_factor = float(np.mean(clean_flux[norm_range_start:norm_range_end]))
    else:
        norm_factor = float(np.mean(clean_flux))
    if norm_factor == 0.0:
        norm_factor = 1.0
    return clean_flux / norm_factor, error / norm_factor


def sample_indices(total_count: int, n_samples: int, seed: int) -> List[int]:
    if total_count < n_samples:
        raise ValueError(f"Requested {n_samples} samples, but source only has {total_count} items.")
    rng = random.Random(seed)
    return rng.sample(range(total_count), n_samples)


def build_from_pth(pth_path: str, output_path: str, n_samples: int, seed: int = 42) -> None:
    data = torch.load(pth_path, map_location="cpu")
    required_keys = {"wave", "clean_flux", "error"}
    if not isinstance(data, dict) or not required_keys.issubset(data.keys()):
        raise ValueError(f"Unsupported pth format in {pth_path}")

    total_count = int(data["clean_flux"].shape[0])
    indices = sample_indices(total_count, n_samples, seed)

    wave_arr = data["wave"][indices].cpu().numpy().astype(np.float32)
    clean_arr = data["clean_flux"][indices].cpu().numpy().astype(np.float32)
    error_arr = data["error"][indices].cpu().numpy().astype(np.float32)
    sample_id_arr = np.array([f"cached_idx_{idx}" for idx in indices], dtype=h5py.string_dtype(encoding="utf-8"))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("wave", data=wave_arr, compression="gzip")
        handle.create_dataset("clean_flux", data=clean_arr, compression="gzip")
        handle.create_dataset("error", data=error_arr, compression="gzip")
        handle.create_dataset("sample_id", data=sample_id_arr, compression="gzip")
        handle.create_dataset("source_index", data=np.array(indices, dtype=np.int64), compression="gzip")
        handle.attrs["n_samples"] = int(clean_arr.shape[0])
        handle.attrs["spectrum_length"] = int(clean_arr.shape[1])
        handle.attrs["normalization"] = "already normalized in cached_data/train.pth using mean(clean_flux[1200:1300])"
        handle.attrs["source_type"] = "pth"
        handle.attrs["source_path"] = pth_path
        handle.attrs["random_seed"] = int(seed)

    print(f"Saved HDF5 subset to {output_path}")
    print(f"Samples: {clean_arr.shape[0]}")
    print(f"Spectrum length: {clean_arr.shape[1]}")


def build_from_out_files(input_glob: str, output_path: str, n_samples: int, seed: int = 42) -> None:
    import glob

    spectrum_files = glob.glob(input_glob)
    if not spectrum_files:
        raise FileNotFoundError(f"No spectrum files found for glob pattern: {input_glob}")
    if len(spectrum_files) < n_samples:
        raise ValueError(f"Requested {n_samples} samples, but only {len(spectrum_files)} files are available.")

    rng = random.Random(seed)
    rng.shuffle(spectrum_files)
    selected_files = spectrum_files[:n_samples]

    waves = []
    clean_fluxes = []
    errors = []
    sample_ids = []

    for file_path in selected_files:
        flux_ = np.genfromtxt(file_path)
        if flux_.ndim < 2 or flux_.shape[1] < 3:
            continue

        wave_ = flux_[:, 0].astype(np.float32)
        clean_flux_ = flux_[:, 1].astype(np.float32)
        error_ = flux_[:, 2].astype(np.float32)
        clean_flux_, error_ = normalize_spectrum(clean_flux_, error_)

        waves.append(wave_)
        clean_fluxes.append(clean_flux_.astype(np.float32))
        errors.append(error_.astype(np.float32))
        sample_ids.append(os.path.basename(file_path))

    if not clean_fluxes:
        raise ValueError("No valid spectra could be loaded from the selected files.")

    min_len = min(len(wave_) for wave_ in waves)
    wave_arr = np.stack([wave_[:min_len] for wave_ in waves], axis=0)
    clean_arr = np.stack([clean_flux_[:min_len] for clean_flux_ in clean_fluxes], axis=0)
    error_arr = np.stack([error_[:min_len] for error_ in errors], axis=0)
    sample_id_arr = np.array(sample_ids, dtype=h5py.string_dtype(encoding="utf-8"))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("wave", data=wave_arr, compression="gzip")
        handle.create_dataset("clean_flux", data=clean_arr, compression="gzip")
        handle.create_dataset("error", data=error_arr, compression="gzip")
        handle.create_dataset("sample_id", data=sample_id_arr, compression="gzip")
        handle.attrs["n_samples"] = int(clean_arr.shape[0])
        handle.attrs["spectrum_length"] = int(clean_arr.shape[1])
        handle.attrs["normalization"] = "mean(clean_flux[1200:1300])"
        handle.attrs["source_type"] = "out_glob"
        handle.attrs["source_path"] = input_glob
        handle.attrs["random_seed"] = int(seed)

    print(f"Saved HDF5 subset to {output_path}")
    print(f"Samples: {clean_arr.shape[0]}")
    print(f"Spectrum length: {clean_arr.shape[1]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a random 90,000-sample HDF5 subset from training data.")
    parser.add_argument("--input-pth", default="/workspace/cached_data/train.pth", help="Input cached train .pth path")
    parser.add_argument("--glob", default=None, help="Optional raw spectrum glob pattern. If set, use raw .out files instead of pth")
    parser.add_argument("--output", default="/workspace/git/train_90000_random.h5", help="Output HDF5 path")
    parser.add_argument("--n-samples", type=int, default=90000, help="Number of randomly selected spectra")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.glob:
        build_from_out_files(args.glob, args.output, args.n_samples, args.seed)
    else:
        build_from_pth(args.input_pth, args.output, args.n_samples, args.seed)


if __name__ == "__main__":
    main()
