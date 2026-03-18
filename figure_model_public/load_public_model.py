"""Example loader for the published Figure checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from published_model import checkpoint_summary, denoise_flux, load_model, load_spectrum_file


DEFAULT_SAMPLE = Path("/workspace/DATA/Test")


def main() -> None:
    model = load_model()
    summary = checkpoint_summary()
    print("Loaded checkpoint")
    print(f"epoch: {summary['epoch']}")
    print(f"global_step: {summary['global_step']}")
    print(f"device: {next(model.parameters()).device}")

    sample_files = sorted(DEFAULT_SAMPLE.glob("*.out"))
    if not sample_files:
        print("No sample spectra found in /workspace/DATA/Test")
        return

    sample = load_spectrum_file(sample_files[0])
    clean_flux = sample["clean_flux"]
    error = sample["error"]
    sn = 10.0
    noise = np.random.default_rng(42).normal(0.0, np.clip(error / sn, 1e-9, None)).astype(np.float32)
    noisy_flux = clean_flux + noise
    denoised_flux = denoise_flux(model, noisy_flux)

    print(f"sample_file: {sample_files[0]}")
    print(f"input_shape: {tuple(noisy_flux.shape)}")
    print(f"output_shape: {tuple(denoised_flux.shape)}")
    print(f"input_mean: {float(noisy_flux.mean()):.6f}")
    print(f"output_mean: {float(denoised_flux.mean()):.6f}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
