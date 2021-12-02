from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.random import Generator, default_rng

_DEFAULT_RNG = default_rng()


def get_rng(seed: int) -> Generator:
    return np.random.Generator(np.random.PCG64(seed))


def generate_clock(
    length: int,
    n_series: int,
) -> np.ndarray:
    return np.tile(np.arange(length), (n_series, 1))


def generate_synthetic_data(
    base_clock: np.ndarray,
    *,
    shift: float = 0,
    period: float = 1,
    amplitude: float = 0,
    phase: float = 0,
    linear_increment: float = 0,
    noise_std: float = 0,
    rng: Generator = _DEFAULT_RNG,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic data with a linear and sinusoidal component.
    """
    feature = np.full(base_clock.shape, shift, dtype=np.float64)
    label = np.zeros(feature.shape, dtype=np.bool_)

    if amplitude:
        feature += amplitude * np.sin(phase + 2 * np.pi / period * base_clock)
        label[:, :-1] = feature[:, 1:] >= feature[:, :-1]
        label[:, -1] = label[:, -2]
    else:
        raise ValueError

    if linear_increment:
        feature += linear_increment * base_clock

    if noise_std:
        feature += rng.normal(scale=noise_std, size=base_clock.shape)

    return feature, label


def create_synthetic_dataframe(
    *,
    length: int,
    n_series: int,
    shift: float = 0,
    period: float = 1,
    amplitude: float = 0,
    phase: float = 0,
    linear_increment: float = 0,
    noise_std: float = 0,
    rng: Generator = _DEFAULT_RNG,
) -> pd.DataFrame:
    id_ = np.repeat(np.arange(n_series), length)
    clock = generate_clock(length, n_series)
    feature, label = generate_synthetic_data(
        clock,
        shift=shift,
        period=period,
        amplitude=amplitude,
        phase=phase,
        linear_increment=linear_increment,
        noise_std=noise_std,
        rng=rng,
    )
    return pd.DataFrame(
        dict(
            id=id_,
            time=clock.ravel(),
            feature=feature.ravel(),
            label=label.ravel(),
        )
    )