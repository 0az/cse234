from models import MLP
from data_generation import create_synthetic_dataframe
from pyspark.sql import SparkSession


def main():
    spark = SparkSession.builder.getOrCreate()

    # df = (
    #     spark.read.option("header", True)
    #     .option("inferSchema", True)
    #     .parquet("./data/dataTraining.parquet")
    # )
    df = create_synthetic_dataframe(
        length=10,
        n_series=1000,
        shift=0,
        period=10,
        amplitude=1,
        phase=0,
        linear_increment=0.02,
        noise_std=0.2,
    )

    df = spark.createDataFrame(df)

    print(df.columns)

    model = MLP(
        params={'window_size': [3, 4, 5, 6], "sample_strategy": ['boolean']}, 
        dataset=df,
        spark=spark,
        time_col="time",
        value_col="feature",
        label_col="label",
        num_workers=4
    )
    model.train(epochs=1)


if __name__ == "__main__":
    main()
