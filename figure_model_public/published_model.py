"""Inference-only model definition for the Figure results checkpoint."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


DEFAULT_CHECKPOINT_PATH = Path("/workspace/Figure/model/final_model.ckpt")


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 20000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"Input seq len {x.size(1)} exceeds max_len {self.pe.size(1)}")
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x, x, x)[0]))
        x = self.norm2(x + self.ffn(x))
        return x


class EnhancedUNetTransformer(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        enc_ch1: int = 96,
        enc_ch2: int = 192,
        d_model: int = 384,
        nhead: int = 12,
        num_transformer_layers: int = 8,
        dim_feedforward: int = 1536,
        dec_ch2: int = 192,
        dec_ch1: int = 96,
        output_projection_channels: int = 64,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv1d(input_channels, enc_ch1, 7, padding=3), nn.InstanceNorm1d(enc_ch1), nn.ReLU())
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = nn.Sequential(nn.Conv1d(enc_ch1, enc_ch2, 7, padding=3), nn.InstanceNorm1d(enc_ch2), nn.ReLU())
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = nn.Sequential(nn.Conv1d(enc_ch2, d_model, 7, padding=3), nn.InstanceNorm1d(d_model), nn.ReLU())
        self.pool3 = nn.MaxPool1d(2)

        self.pos_encoding = PositionalEncoding1D(d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_transformer_layers)]
        )

        self.up3_conv = nn.Sequential(nn.Conv1d(d_model, dec_ch2, 3, padding=1), nn.InstanceNorm1d(dec_ch2), nn.ReLU())
        self.dec3 = nn.Sequential(nn.Conv1d(dec_ch2 + d_model, dec_ch2, 3, padding=1), nn.InstanceNorm1d(dec_ch2), nn.ReLU())
        self.up2_conv = nn.Sequential(nn.Conv1d(dec_ch2, dec_ch1, 3, padding=1), nn.InstanceNorm1d(dec_ch1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Conv1d(dec_ch1 + enc_ch2, dec_ch1, 3, padding=1), nn.InstanceNorm1d(dec_ch1), nn.ReLU())
        self.up1_conv = nn.Sequential(nn.Conv1d(dec_ch1, output_projection_channels, 3, padding=1), nn.InstanceNorm1d(output_projection_channels), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv1d(output_projection_channels + enc_ch1, output_projection_channels, 3, padding=1), nn.InstanceNorm1d(output_projection_channels), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        t = p3.permute(0, 2, 1)
        t = self.pos_encoding(t)
        for block in self.transformer_blocks:
            t = block(t)
        t = t.permute(0, 2, 1)

        u3 = F.interpolate(t, size=e3.shape[-1], mode="linear", align_corners=False)
        d3 = self.dec3(torch.cat([self.up3_conv(u3), e3], dim=1))
        u2 = F.interpolate(d3, size=e2.shape[-1], mode="linear", align_corners=False)
        d2 = self.dec2(torch.cat([self.up2_conv(u2), e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-1], mode="linear", align_corners=False)
        d1 = self.dec1(torch.cat([self.up1_conv(u1), e1], dim=1))
        return d1.permute(0, 2, 1)


class PublishedDenoisingModel(pl.LightningModule):
    def __init__(
        self,
        enc_ch1: int = 96,
        enc_ch2: int = 192,
        d_model: int = 384,
        nhead: int = 12,
        num_transformer_layers: int = 8,
        dim_feedforward: int = 1536,
        dec_ch2: int = 192,
        dec_ch1: int = 96,
        core_output_channels: int = 64,
        dropout: float = 0.15,
        **unused_hparams: Any,
    ) -> None:
        super().__init__()
        # Accept extra checkpoint hyperparameters for compatibility,
        # but keep the public inference API limited to architecture settings.
        del unused_hparams
        self.model_core = EnhancedUNetTransformer(
            enc_ch1=enc_ch1,
            enc_ch2=enc_ch2,
            d_model=d_model,
            nhead=nhead,
            num_transformer_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward,
            dec_ch2=dec_ch2,
            dec_ch1=dec_ch1,
            output_projection_channels=core_output_channels,
            dropout=dropout,
        )
        self.final_projection = nn.Conv1d(core_output_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        core_output = self.model_core(x)
        return self.final_projection(core_output.permute(0, 2, 1)).permute(0, 2, 1)


def get_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    device: Optional[str] = None,
    eval_mode: bool = True,
) -> PublishedDenoisingModel:
    checkpoint_path = Path(checkpoint_path)
    model = PublishedDenoisingModel.load_from_checkpoint(str(checkpoint_path), strict=False)
    model = model.to(get_device(device))
    if eval_mode:
        model.eval()
    return model


def normalize_flux_and_error(clean_flux: np.ndarray, error: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    norm_range_end = min(1300, len(clean_flux))
    norm_range_start = min(1200, norm_range_end - 1)
    if norm_range_end > norm_range_start:
        norm_factor = float(np.mean(clean_flux[norm_range_start:norm_range_end]))
    else:
        norm_factor = float(np.mean(clean_flux))
    if norm_factor == 0.0:
        norm_factor = 1.0
    return clean_flux / norm_factor, error / norm_factor


@torch.no_grad()
def denoise_flux(model: PublishedDenoisingModel, noisy_flux: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(noisy_flux, np.ndarray):
        tensor = torch.from_numpy(noisy_flux.astype(np.float32))
    else:
        tensor = noisy_flux.float()

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0).unsqueeze(-1)
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    elif tensor.ndim != 3:
        raise ValueError("Expected noisy_flux with shape (L,), (B, L), or (B, L, 1)")

    tensor = tensor.to(next(model.parameters()).device)
    pred = model(tensor).squeeze(-1).cpu().numpy()
    return pred[0] if pred.shape[0] == 1 else pred


def load_spectrum_file(file_path: str | Path) -> Dict[str, np.ndarray]:
    raw = np.genfromtxt(file_path)
    if raw.ndim < 2 or raw.shape[1] < 3:
        raise ValueError(f"Invalid spectrum file: {file_path}")
    wave = raw[:, 0].astype(np.float32)
    clean_flux = raw[:, 1].astype(np.float32)
    error = raw[:, 2].astype(np.float32)
    clean_flux, error = normalize_flux_and_error(clean_flux, error)
    return {"wave": wave, "clean_flux": clean_flux, "error": error}


def checkpoint_summary(checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    return {
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "hyper_parameters": ckpt.get("hyper_parameters", {}),
    }
