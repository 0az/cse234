from pyspark.sql import SparkSession
from pyspark.sql.functions import array_repeat, col, explode, lit, randn

from .generate import create_synthetic_dataframe
from .logs import get_logger
from .models import MLP
from .timing import Timer

LOGGER = get_logger(__name__)


class Args:
    size: int
    workers: int
    data_path: str


class ExperimentArgs(Args):
    store_type: str
    store_path: str
    grid_preset: str


class GenerateArgs(Args):
    pass


PARAMS = {
    'window_size': [3, 5],
    'batch_size': [8],
    'sample_strategy': ['boolean'],
    'num_hidden_layers': [0, 1],
    'final_activation': ['relu', 'sigmoid'],
}
# 2 meta, 6 hyper
SMALL_GRID = PARAMS.copy()
SMALL_GRID.update(
    batch_size=[8],
    num_hidden_layers=[0, 1, 2],
)
# 2 meta, 40 hyper
LARGE_GRID = PARAMS.copy()
LARGE_GRID.update(
    batch_size=[200],
    lr=[0.01, 0.005, 0.001, 0.0005, 0.0001],
    momentum=[0.9, 0.8],
)


def generate(args: GenerateArgs):
    if not args.data_path.startswith('hdfs://'):
        raise ValueError('Output path must be on HDFS')

    LOGGER.info('---')
    LOGGER.info('config:')
    LOGGER.info('  size: %d', args.size)
    LOGGER.info('  workers: %d', args.workers)
    LOGGER.info('  data_path: %s', args.data_path)

    timer = Timer()
    timer.start()

    LOGGER.debug('Starting generation')
    timer.split('generation')

    LOGGER.info('SparkSession: Init starting')
    spark = SparkSession.builder.getOrCreate()
    LOGGER.info('SparkSession: Init complete')

    LOGGER.info('Creating Pandas DataFrame')
    timer.split('pandas df')
    # Create a ~1mb DF
    # ~25b/row means a length=40 series is ~1kb
    # Thus, n = 1000 is ~1mb
    pandas_df = create_synthetic_dataframe(
        length=40,
        n_series=1000,
        shift=0,
        period=10,
        amplitude=1,
        phase=0,
        linear_increment=0.02,
        # Relatively low noise, as we'll add this noise once again in Spark
        # Adding 0.2 std gaussian noise twice results in ~0.28 std noise
        noise_std=0.2,
    )
    timer.split('pandas df')

    LOGGER.info('Creating and exploding Spark DataFrame')
    timer.split('spark df prep')
    df = spark.createDataFrame(pandas_df).cache()
    df = df.withColumn('_void', explode(array_repeat(lit(None), args.size)))
    df = df.drop('_void')
    df = df.select(
        col('id'),
        col('time'),
        (col('feature') + randn()).alias('feature'),
        col('label'),
        (col('id') / 100).alias('partition')
    )
    timer.split('spark df prep')

    LOGGER.info('Serializing Spark DataFrame')
    timer.split('spark df write')
    df.write.parquet(args.data_path, partitionBy='partition')
    timer.split('spark df write')
    LOGGER.info('Finished serialization')

    timer.split('generation')
    timer.print_splits()

    LOGGER.info('Finished generation')


def experiment(args: ExperimentArgs):
    if args.size < 1:
        raise ValueError('Size must be at least 1, got %d' % args.size)

    LOGGER.info('Starting experiment')
    timer = Timer()
    timer.start()
    timer.split('experiment')

    LOGGER.info('---')
    LOGGER.info('config:')
    LOGGER.info('  size: %d', args.size)
    LOGGER.info('  workers: %d', args.workers)
    LOGGER.info('  data_path: %s', args.data_path)

    LOGGER.info('SparkSession: Init starting')
    spark = SparkSession.builder.getOrCreate()
    LOGGER.info('SparkSession: Init complete')

    LOGGER.info('Loading data')
    timer.split('data load')
    df = spark.read.parquet(args.data_path)
    timer.split('data load')

    LOGGER.info('Initializing MLP Model')
    timer.split('model init')

    params = SMALL_GRID if args.grid_preset == 'small' else LARGE_GRID
    model = MLP(
        params=params,
        dataset=df,
        spark=spark,
        time_col='time',
        value_col='feature',
        label_col='label',
        n_workers=args.workers,
        store_type=args.store_type,
        store_path=args.store_path,
    )
    timer.split('model init')

    LOGGER.info('Training model')
    timer.split('model train')
    model.train(epochs=1)
    timer.split('model train')
    LOGGER.info('Training complete')
    timer.split('experiment')

    timer.print_splits(LOGGER.print)

    LOGGER.info('Completed experiment')
