from pyspark.sql import SparkSession
from pyspark.sql.functions import array_repeat, explode, lit

from .generate import create_synthetic_dataframe
from .logs import get_logger
from .models import MLP

LOGGER = get_logger(__name__)


class Args:
    size: int


def main(args: Args):
    if args.size < 1:
        raise ValueError('Size must be at least 1, got %d' % args.size)

    LOGGER.debug('Starting main')

    LOGGER.info('SparkSession: Init starting')
    spark = SparkSession.builder.getOrCreate()
    LOGGER.info('SparkSession: Init complete')

    LOGGER.info('Creating Pandas DataFrame')
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

    LOGGER.info('Creating and exploding Spark DataFrame')
    df = spark.createDataFrame(pandas_df).cache()
    df = df.withColumn('_void', explode(array_repeat(lit(None), args.size)))
    df = df.drop('_void')

    LOGGER.info('Serializing Spark DataFrame')
    df = df.persist()

    LOGGER.info('Initializing MLP Model')
    model = MLP(
        params={'window_size': [3, 4], "sample_strategy": ['boolean']},
        dataset=df,
        spark=spark,
        time_col="time",
        value_col="feature",
        label_col="label",
        num_workers=2,
    )

    LOGGER.info('Training model')
    model.train(epochs=1)
    LOGGER.info('Training complete')
