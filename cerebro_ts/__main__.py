import argparse

from cerebro_ts.utils import positive_int

ap = argparse.ArgumentParser()

def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--workers',
        '-w',
        type=positive_int,
        default=1,
        help='Number of Cerebro worker instances',
    )
    parser.add_argument(
        '--size',
        '-s',
        type=positive_int,
        default=1,
        help='Approximate dataset size in MB',
    )
    parser.add_argument(
        '--data-path',
        '-p',
        required=True,
        help='Dataset path',
    )

sp = ap.add_subparsers(title='Commands', dest='command')

xp = sp.add_parser(
    'experiment',
    help='Run an experiment on a provided dataset',
)
gp = sp.add_parser(
    'generate',
    help='Generate a synthetic dataset',
)

add_common_args(xp)
add_common_args(gp)

xp.add_argument(
    '--grid-preset',
    '-g',
    choices=('small', 'large'),
    default='small',
    help='Hyperparameter grid preset',
)

store_type = xp.add_mutually_exclusive_group()
store_type.add_argument(
    '--local',
    action='store_const',
    const='local',
    dest='store_type',
    help='Use the LocalStore Cerebro storage driver',
)
store_type.add_argument(
    '--hdfs',
    action='store_const',
    const='hdfs',
    dest='store_type',
    help='Use the HDFSStore Cerebro storage driver',
)
store_type.set_defaults(store_type='local')

xp.add_argument(
    '--store-path',
    default='/tmp/cerebro',
    help='Location of the Cerebro data store',
)

args = ap.parse_args()


# Lazy imports

from cerebro_ts.logs import init_logging

# Needs to happen before main is imported
init_logging()

from cerebro_ts.main import experiment, generate

if args.command == 'generate':
    generate(args)
elif args.command == 'experiment':
    experiment(args)  # type: ignore
