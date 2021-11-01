from types import LambdaType
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from petastorm.tf_utils import make_reader
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F


def prep_data(source, window_size, sampling_factor, sample_strategy):
    """Pulls data from source and preps it into windows given strategy.
    
    Parameters
    ----------
    source: path to petastorm time series dataset
    window_size: size of window to use
    sampling_factor: factor to downsample by
    sample_strategy: strategy to use for sampling. One of 'mean' or 'boolean'  

    Returns
    -------
    windows: list of windows
    """
    with make_reader(source) as reader:
        dataset = make_reader(reader)
        X = np.fromiter(map(lambda x: x[0], dataset), dtype=np.float32)
        y = np.fromiter(map(lambda x: x[1], dataset), dtype=np.float32)
        windowed_ds = keras.utils.timeseries_dataset_from_array(X, y, window_size, sampling_rate = sampling_factor)
    
    return windowed_ds

# TODO: how to get this to work with tensorflow? We could write these 
# parquet files to disk (for each window size) and then read them in,
# but that's not the best way to do it.
def prep_data_spark(spark, dataset_url, window_size, sampling_factor, sample_strategy):
    """
    Windows data that is already in a spark dataframe.

    Parameters
    ----------
    spark: spark session
    dataset_url: str, path to parquet file
    window_size: size of window to use
    sampling_factor: factor to downsample by
    sample_strategy: strategy to use for sampling. One of 'mean' or 'boolean'  

    Returns
    -------
    windows: spark.sql.DataFrame containing windows
    """
    TIME_COL = 'time'
    VALUE_COL = 'value'
    LABEL_COL = 'label'

    df = spark.read.parquet(dataset_url)
    df = df.withColumn(TIME_COL, df[TIME_COL].cast('timestamp'))

    # apply windowing to (time, value) pairs
    windowSpec = Window.partitionBy(LABEL_COL).orderBy(F.col(TIME_COL).desc())
    windowed_df = df.over(windowSpec)

    return windowed_df