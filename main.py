from models import MLP
from pyspark.sql import SparkSession


def main():
    spark = SparkSession.builder.getOrCreate()

    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .parquet("./data/dataTraining.parquet")
    )

    model = MLP(
        params={'window_size': [3, 4, 5, 6], "sample_strategy": ['mean', 'boolean']}, 
        dataset=df,
        spark=spark,
        time_col="date",
        value_col="Temperature",
        label_col="Occupancy",
        num_workers=4
    )
    model.train(epochs=1)


if __name__ == "__main__":
    main()
