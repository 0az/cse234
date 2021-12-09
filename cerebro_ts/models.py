import os
from itertools import product

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
from joblib import Parallel, delayed
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from .logs import get_logger
from .prepare import prep_data_spark

LOGGER = get_logger(__name__)


class Model:
    DATAPREP_PARAMS = {"window_size", "sample_strategy"}
    DEFAULTS = {
        "window_size": [1],
        "sample_strategy": ["mean"],
        "lr": [0.001],
        "momentum": [0.9],
        "batch_size": [8],
        "num_hidden_layers": [0],
        "hidden_dim": [32],
        "hidden_activation": ["relu"],
        "final_activation": ["sigmoid"],
    }

    def __init__(
        self,
        params,
        dataset,
        spark,
        cache_prepped_data: bool = False,
        n_workers: int = 1,
        time_col: str = "time",
        label_col: str = "label",
        id_col: str = "id",
        value_col: str = "value",
        store_type: str = 'local',
        store_path: str = '/tmp/cerebro',
    ):
        """
        Dataset must be in form of (id, timestamp, value, label)
        """
        self.dataprep_params = {}
        self.params = {}
        # ensure defaults are in dict
        for d in Model.DEFAULTS:
            if d not in params:
                params[d] = Model.DEFAULTS[d]
        # separate dataprep params from cerebro params
        for p in params:
            if p in Model.DATAPREP_PARAMS:
                self.dataprep_params[p] = params[p]
                # if p == 'window_size':
                #     # do the +1 because we pass in the window columns +  the original column
                #     self.params['input_dim'] = hp_choice(list(map(lambda x: x + 1, params[p])))
            else:
                self.params[p] = (
                    hp_choice(params[p]) if type(params[p]) == list else params[p]
                )

        valid_cols = set(dataset.columns).intersection(
            set([id_col, time_col, value_col, label_col])
        )
        self.dataset = dataset.select(*valid_cols)
        self.spark = spark
        self.cache_prepped_data = cache_prepped_data
        self.num_workers = num_workers

        self.input_dim = None

        # define columns
        self.id_col = id_col
        self.time_col = time_col
        self.value_col = value_col
        self.label_col = label_col

        self.best_model = None
        self.best_acc = float("-inf")
        self.best_w = None

        # Create cerebro basics
        self.backend = SparkBackend(
            spark_context=self.spark.sparkContext, num_workers=self.num_workers
        )
        self.store = LocalStore(prefix_path=os.path.join(os.getcwd(), "cerebro_store"))

        if store_type == 'local':
            self.store = LocalStore(prefix_path=store_path)
        elif store_type == 'hdfs':
            self.store = HDFSStore(prefix_path=store_path)
        else:
            raise ValueError(
                'Invalid store type: expected local or hdfs,'
                f' got {store_type}'
            )

        if self.cache_prepped_data:
            # compute the dataframes now and cache them
            pass

    def estimator_gen_fn(self, params):
        raise NotImplementedError("Model function not implemented.")

    def train(self, epochs=10):
        """Trains model on pre-loaded data."""
        if not self.cache_prepped_data:
            for w, s in product(
                self.dataprep_params["window_size"],
                self.dataprep_params["sample_strategy"],
            ):
                self.input_dim = w + 1
                model_selection = GridSearch(
                    self.backend,
                    self.store,
                    self.estimator_gen_fn,
                    search_space=self.params,
                    num_epochs=epochs,
                    evaluation_metric="loss",
                    label_columns=[self.label_col],
                    feature_columns=["features"],
                    verbose=1,
                )
                prepped = prep_data_spark(
                    self.dataset,
                    w,
                    1,
                    s,
                    time_col=self.time_col,
                    label_col=self.label_col,
                    id_col=self.id_col,
                    value_col=self.value_col,
                )

                model = model_selection.fit(prepped)
                print(model.get_history())
                if model.get_history()["val_acc"][-1] > self.best_acc:
                    LOGGER.info(f'Found new best model with window size {w}')
                    self.best_acc = model.get_history()["val_acc"][-1]
                    self.best_model = model.keras()
                    self.best_w = w
        

        LOGGER.print('MLP Training complete:')
        LOGGER.print(f'\tValidation accuracy: {self.best_acc}')
        LOGGER.print(f'\tWindow size: {self.best_w}')

        # task = delayed(self._train_one)

        # Parallel(n_jobs=len(self.dataprep_params["window_size"]))(task(w) for w in self.dataprep_params["window_size"])

    def predict(self, X):
        """Predicts on given data."""
        if self.best_model is None:
            raise ValueError("Model not trained yet.")

        return self.best_model.predict(X)



class MLP(Model):

    def __init__(
        self,
        params,
        dataset,
        spark,
        cache_prepped_data: bool = False,
        n_workers: int = 1,
        time_col: str = "time",
        label_col: str = "label",
        id_col: str = "id",
        value_col: str = "value",
        store_type: str = 'local',
        store_path: str = '/tmp/cerebro',
    ):
        """
        Dataset must be in form of (id, timestamp, value, label)
        """
        super(MLP, self).__init__(
            params,
            dataset,
            spark,
            cache_prepped_data=cache_prepped_data,
            num_workers=n_workers,
            time_col=time_col,
            label_col=label_col,
            id_col=id_col,
            value_col=value_col,
            store_type=store_type,
            store_path=store_path,
        )

    def estimator_gen_fn(self, params):
        model = tf.keras.models.Sequential()
        input_dim = self.input_dim
        model.add(tf.keras.layers.Input(shape=input_dim, name="features"))
        for i in range(params["num_hidden_layers"]):
            model.add(
                tf.keras.layers.Dense(
                    params["hidden_dim"],
                    input_dim=input_dim if i == 0 else params["hidden_dim"],
                    activation=params["hidden_activation"],
                )
            )
        model.add(
            tf.keras.layers.Dense(
                1,
                input_dim=input_dim
                if params["num_hidden_layers"] == 0
                else params["hidden_dim"],
                activation=params["final_activation"],
            )
        )

        optimizer = tf.keras.optimizers.SGD(
            lr=params["lr"], momentum=params["momentum"]
        )
        loss = "mse"

        estimator = SparkEstimator(
            model=model,
            optimizer=optimizer,
            loss=loss,
            metrics=["acc"],
            batch_size=params["batch_size"],
        )

        return estimator

    # def _train_one(self, w):
    #     """Train one GridSearch task given a window size."""
    #     self.input_dim = w + 1
    #     model_selection = GridSearch(
    #         self.backend,
    #         self.store,
    #         self.estimator_gen_fn,
    #         search_space=self.params,
    #         num_epochs=epochs,
    #         evaluation_metric="loss",
    #         label_columns=[self.label_col],
    #         feature_columns=["features"],
    #         verbose=1,
    #     )
    #     prepped = prep_data_spark(
    #         self.dataset,
    #         w,
    #         1,
    #         "mean",
    #         time_col=self.time_col,
    #         label_col=self.label_col,
    #         id_col=self.id_col,
    #         value_col=self.value_col,
    #     ).repartition(self.num_workers)

    #     model = model_selection.fit(prepped)

    #     return (model.get_history()['val_acc'][0], model.keras())


class LSTM(Model):
    def __init__(
        self,
        params,
        dataset,
        spark,
        cache_prepped_data: bool = False,
        n_workers: int = 1,
        time_col: str = "time",
        label_col: str = "label",
        id_col: str = "id",
        value_col: str = "value",
        store_type: str = 'local',
        store_path: str = '/tmp/cerebro',
    ):
        """
        Dataset must be in form of (id, timestamp, value, label)
        """
        super(LSTM, self).__init__(
            params,
            dataset,
            spark,
            cache_prepped_data=cache_prepped_data,
            num_workers=n_workers,
            time_col=time_col,
            label_col=label_col,
            id_col=id_col,
            value_col=value_col,
            store_type=store_type,
            store_path=store_path,
        )

    def estimator_gen_fn(self, params):
        model = tf.keras.models.Sequential()
        input_dim = self.input_dim
        model.add(tf.keras.layers.Input(shape=input_dim, name="features"))
        model.add(tf.keras.layers.Reshape((1, -1)))
        if params["num_hidden_layers"] > 1:
            for i in range(params['num_hidden_layers']):
                model.add(tf.keras.layers.LSTM(params['hidden_dim'], return_sequences=True, name=f'lstm{i}'))
        model.add(
            tf.keras.layers.LSTM(
                params["hidden_dim"], activation=params["final_activation"]
            )
        )

        optimizer = tf.keras.optimizers.SGD(
            lr=params["lr"], momentum=params["momentum"]
        )
        loss = "mse"

        estimator = SparkEstimator(
            model=model,
            optimizer=optimizer,
            loss=loss,
            metrics=["acc"],
            batch_size=params["batch_size"],
        )

        return estimator