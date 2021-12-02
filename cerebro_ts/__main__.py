import argparse

from cerebro_ts.utils import positive_int

ap = argparse.ArgumentParser()
ap.add_argument(
    '--size',
    '-s',
    type=positive_int,
    default=1,
    help='Approximate dataset size in MB',
)
args = ap.parse_args()


# Lazy imports
from cerebro_ts.logging import init_logging
from cerebro_ts.main import main

init_logging()
main(args)  # type: ignore
