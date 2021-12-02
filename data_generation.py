import argparse
from pathlib import Path
from typing import Any, NamedTuple, Tuple

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


if __name__ == '__main__':
    BYTES_PER_ROW = 25

    def positive_int(s: str) -> int:
        i = int(s)
        if i <= 0:
            raise ValueError(f'Expected positive integer, got {s}')
        return i

    class default(NamedTuple):
        value: Any = None

    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--difficulty', choices=['easy', 'medium', 'hard'], default='easy'
    )
    ap.add_argument(
        '--size',
        type=int,
        default=-1,
        help='The approximate dataset size in MB',
    )
    ap.add_argument('--series-length', type=positive_int, default=25)
    ap.add_argument('--series-count', type=positive_int, default=default(100))
    ap.add_argument('--period', type=positive_int, default=10)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()

    if args.size != -1:
        if not isinstance(args.series_count, default):
            raise ValueError('Cannot specify size and series count')
        args.series_count = int(
            args.size * 1e6 // BYTES_PER_ROW // args.series_length
        )
    elif isinstance(args.series_count, default):
        args.series_count = args.series_count.value

    _base = dict(
        length=args.series_length,
        n_series=args.series_count,
        shift=0,
        period=args.period,
        amplitude=1,
        phase=0,
        linear_increment=0,
        noise_std=0,
    )
    if args.series_length <= args.period:
        raise ValueError(
            f'Series length is less than period ({args.period}),'
            ' which may lead to poor results'
        )

    difficulty = {
        'easy': {},
        'medium': dict(
            linear_increment=0.01,
            noise_std=0.2,
        ),
        'hard': dict(
            linear_increment=0.1,
            noise_std=0.4,
        ),
    }
    for level, d in difficulty.items():
        difficulty[level] = _base | d

    df = create_synthetic_dataframe(**difficulty[args.difficulty])

    df.to_parquet(str(args.output), index=False)
