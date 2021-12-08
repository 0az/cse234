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

from cerebro_ts.logs import init_logging

# Needs to happen before main is imported
init_logging()

from cerebro_ts.main import main

main(args)  # type: ignore
