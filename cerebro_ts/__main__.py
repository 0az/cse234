import argparse

from .utils import positive_int

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
from .logging import init_logging
from .main import main

init_logging()
main(args)  # type: ignore
