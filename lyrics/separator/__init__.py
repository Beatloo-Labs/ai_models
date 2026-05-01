"""Minimal vocal-separation runtime.

Replaces the relevant slice of MSST `utils/settings.py` + `utils/model_utils.py`
needed to run anvuew BS-Roformer for vocals extraction. Vendored to keep
self-contained — no MSST checkout required on the target machine.
"""
from pathlib import Path
from typing import Dict

import yaml
import numpy as np
import torch
from torch import nn
from ml_collections import ConfigDict
from tqdm.auto import tqdm

from .bs_roformer import BSRoformer

_HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = _HERE / "config_anvuew_fast.yaml"


def load_config(config_path: str | Path) -> ConfigDict:
    with open(config_path, "r", encoding="utf-8") as f:
        return ConfigDict(yaml.load(f, Loader=yaml.FullLoader))


def load_separator(ckpt_path: str | Path,
                   config_path: str | Path = DEFAULT_CONFIG,
                   device: str = "cuda") -> tuple[nn.Module, ConfigDict]:
    """Build BSRoformer from config, load checkpoint, move to device."""
    cfg = load_config(config_path)
    model = BSRoformer(**dict(cfg.model))
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for k in ("state_dict", "state", "model_state_dict"):
            if k in state:
                state = state[k]
                break
    model.load_state_dict(state)
    return model.to(device).eval(), cfg


def _windowing(window_size: int, fade_size: int) -> torch.Tensor:
    w = torch.ones(window_size)
    w[:fade_size] = torch.linspace(0, 1, fade_size)
    w[-fade_size:] = torch.linspace(1, 0, fade_size)
    return w


def _target_instruments(cfg: ConfigDict):
    target = getattr(cfg.training, "target_instrument", None)
    return [target] if target else list(cfg.training.instruments)


def demix(cfg: ConfigDict, model: nn.Module, mix: torch.Tensor,
          device: str, pbar: bool = False) -> Dict[str, np.ndarray]:
    """Chunked overlap-add inference. Mirrors MSST's generic-mode demix.

    `mix` is shape (channels, samples). Returns {instrument: np.ndarray}.
    """
    if not isinstance(mix, torch.Tensor):
        mix = torch.as_tensor(mix, dtype=torch.float32)
    else:
        mix = mix.detach().to(torch.float32)

    chunk_size = cfg.inference.get("chunk_size", cfg.audio.chunk_size)
    num_overlap = cfg.inference.num_overlap
    batch_size = cfg.inference.batch_size
    fade_size = chunk_size // 10
    step = chunk_size // num_overlap
    border = chunk_size - step
    length_init = mix.shape[-1]
    instruments = _target_instruments(cfg)
    num_inst = len(instruments)
    window = _windowing(chunk_size, fade_size)

    if length_init > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode="reflect")

    use_amp = bool(getattr(cfg.training, "use_amp", True)) and device.startswith("cuda")
    result = torch.zeros((num_inst,) + mix.shape, dtype=torch.float32)
    counter = torch.zeros_like(result)

    batch_data, batch_locs = [], []
    progress = tqdm(total=mix.shape[1], desc="demix", leave=False) if pbar else None

    autocast = torch.amp.autocast("cuda", enabled=use_amp) if device.startswith("cuda") \
        else torch.amp.autocast("cpu", enabled=False)

    with autocast, torch.inference_mode():
        i = 0
        while i < mix.shape[1]:
            part = mix[:, i:i + chunk_size].to(device)
            chunk_len = part.shape[-1]
            pad_mode = "reflect" if chunk_len > chunk_size // 2 else "constant"
            part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode=pad_mode)
            batch_data.append(part)
            batch_locs.append((i, chunk_len))
            i += step

            if len(batch_data) >= batch_size or i >= mix.shape[1]:
                arr = torch.stack(batch_data, dim=0)
                x = model(arr)

                w = window.clone()
                if i - step == 0:
                    w[:fade_size] = 1
                if i >= mix.shape[1]:
                    w[-fade_size:] = 1

                for j, (start, seg_len) in enumerate(batch_locs):
                    result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu() * w[..., :seg_len]
                    counter[..., start:start + seg_len] += w[..., :seg_len]

                batch_data.clear()
                batch_locs.clear()

            if progress:
                progress.update(step)

    if progress:
        progress.close()

    estimated = (result / counter).cpu().numpy()
    np.nan_to_num(estimated, copy=False, nan=0.0)
    if length_init > 2 * border and border > 0:
        estimated = estimated[..., border:-border]
    return {name: estimated[k] for k, name in enumerate(instruments)}
