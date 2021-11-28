import tensorflow as tf
import tensorflow.keras as keras
from cerebro.backend import SparkBackend
from cerebro.keras import SparkEstimator

# datas storage for intermediate data and model artifacts.
from cerebro.storage import HDFSStore, LocalStore

# Utility functions for specifying the search space.
# Model selection/AutoML methods.
from cerebro.tune import (
    GridSearch,
    RandomSearch,
    TPESearch,
    hp_choice,
    hp_loguniform,
    hp_qloguniform,
    hp_quniform,
    hp_uniform,
)
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler


def prep_data_spark(
    spark,
    window_size,
    sampling_factor,
    sample_strategy,
    dataset_path=None,
    id_col="id",
    time_col="time",
    value_col="value",
    label_col="label",
):
    """
    Windows data that is already in a spark dataframe.

    Parameters
    ----------
    spark: spark session or sql dataframe
        Preferred Schema:
        id | time      | value ...  | label
        --------------------------
        int timestamp   float ...
    dataset_path: str, Default None. path to parquet file
    window_size: size of window to use
    label_aggregation_strategy: strategy to use for label aggregation. One of 'mean' or 'boolean'

    Returns
    -------
    windows: spark.sql.DataFrame containing 'feature' and 'label' columns
        'feature' - vector-valued column of windowed data
        'label' - scalar-valued column of labels
    """
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.dataframe import DataFrame
    from pyspark.sql.window import Window

    TIME_COL = "time"
    VALUE_COL = "value"
    LABEL_COL = "label"

    agg_strategies = {"mean": F.mean, "boolean": F.max}

    if isinstance(spark, DataFrame):
        df = spark
    elif isinstance(spark, SparkSession) and dataset_path is not None:
        df = spark.read.parquet(dataset_path)
    else:
        raise TypeError("Expected spark context + filepath or spark.sql.DataFrame")

    # cast timestamp column to timestamp type
    df = df.withColumn(time_col, df[time_col].cast("timestamp"))

    # apply windowing to values pairs
    if id_col in df.columns:
        windowSpec = Window.partitionBy(id_col).orderBy(F.col(time_col).desc())
        windowSpecLabels = (
            Window.partitionBy(id_col)
            .orderBy(F.col(time_col).desc())
            .rowsBetween(-window_size, 0)
        )
    else:
        windowSpec = Window.orderBy(F.col(time_col).desc())
        windowSpecLabels = Window.orderBy(F.col(time_col).desc()).rowsBetween(
            -window_size, 0
        )

    agg_labels = agg_strategies[sample_strategy](label_col).over(windowSpecLabels)

    windowed_df = df.withColumn(f"{value_col}_1", F.lag(value_col, 1).over(windowSpec))

    if window_size > 1:
        for i in range(2, window_size + 1):
            windowed_df = windowed_df.withColumn(
                f"{value_col}_{i}", F.lag(value_col, i).over(windowSpec)
            )

    # remove rows that can't fill a full window and add the labels to the df
    return windowed_df.where(
        F.col(f"{value_col}_{window_size}").isNotNull()
    ).withColumn(label_col, agg_labels)

def estimator_gen_fn(params):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=4, name="features"))
    model.add(tf.keras.layers.Dense(1, input_dim=4))
    model.add(tf.keras.layers.Activation("sigmoid"))

    optimizer = tf.keras.optimizers.SGD(lr=params["lr"])
    loss = "mse"

    estimator = SparkEstimator(
        model=model, optimizer=optimizer, loss=loss, metrics=["acc"], batch_size=8
    )

    return estimator


spark = SparkSession.builder.getOrCreate()

df = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .parquet("./data/dataTraining.parquet")
)


prepped = prep_data_spark(
    df.select("date", "Temperature", "Occupancy"),
    3,
    1,
    "mean",
    time_col="date",
    value_col="Temperature",
    label_col="Occupancy",
)

assembler = VectorAssembler(inputCols=['Temperature', 'Temperature_1', 'Temperature_2', 'Temperature_3'], outputCol='features')
prepped = assembler.transform(prepped)

print("Data Loaded.")

backend = SparkBackend(spark_context=spark.sparkContext, num_workers=2)
store = LocalStore(prefix_path="/Users/arunavgupta/Documents/FA21/cse234/data/")

train_df, test_df = prepped.drop('date').randomSplit([0.8, 0.2])
train_df = train_df.repartition(2)

search_space = {"lr": hp_choice([0.01, 0.001, 0.0001])}

model_selection = GridSearch(
    backend,
    store,
    estimator_gen_fn,
    search_space,
    num_epochs=1,
    evaluation_metric="loss",
    label_columns=["Occupancy"],
    feature_columns=["features"],    
    verbose=1,
)

model = model_selection.fit(train_df)
