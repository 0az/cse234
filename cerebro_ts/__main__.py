import argparse

from cerebro_ts.utils import positive_int

ap = argparse.ArgumentParser()
ap.add_argument(
    '--workers',
    '-w',
    type=positive_int,
    default=1,
    help='Number of Cerebro worker instances',
)
ap.add_argument(
    '--size',
    '-s',
    type=positive_int,
    default=1,
    help='Approximate dataset size in MB',
)
store_type = ap.add_mutually_exclusive_group()
store_type.add_argument(
    '--local',
    action='store_const',
    const='local',
    dest='store_type',
    help='Use the LocalStore Cerebro storage driver.',
)
store_type.add_argument(
    '--hdfs',
    action='store_const',
    const='hdfs',
    dest='store_type',
    help='Use the HDFSStore Cerebro storage driver.',
)
store_type.set_defaults(store_type='local')
ap.add_argument(
    '--store-path',
    default='/tmp/cerebro',
    help='Location of the Cerebro data store.',
)
args = ap.parse_args()


# Lazy imports

from cerebro_ts.logs import init_logging

# Needs to happen before main is imported
init_logging()

from cerebro_ts.main import main

main(args)  # type: ignore
