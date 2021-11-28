import os
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
from collections import subtract

class MLP:

    DATAPREP_PARAMS = {'window_size'}

    def __init__(self, params, dataset, spark, cache_prepped_data=False, num_workers=1):
        self.dataprep_params = {}
        # separate dataprep params from cerebro params
        for p in params:
            if p in MLP.DATAPREP_PARAMS:
                self.dataprep_params[p] = params[p]
                params.pop(p)
        self.params = params
        self.dataset = dataset
        self.spark = spark
        self.cache_prepped_data = cache_prepped_data
        self.num_workers = num_workers

        # Create cerebro primitives
        self.backend = SparkBackend(spark_context=self.spark.sparkContext, num_workers=self.num_workers)
        self.store = LocalStore(prefix_path=os.path.join(os.getcwd(), 'cerebro_store'))

    
    @staticmethod
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
    
    def train(self, epochs=10):
        """Trains model on pre-loaded data."""
        model_selection = GridSearch(
            self.backend,
            self.store,
            MLP.estimator_gen_fn,
            search_space,
            num_epochs=epochs,
            evaluation_metric="loss",
            label_columns=["Occupancy"],
            feature_columns=["features"],    
            verbose=1,
        )

    def predict(self, X):
        """Predicts on given data."""
        pass