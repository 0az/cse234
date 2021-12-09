#! /bin/bash

cd cse234 || exit

# git diff -q && zip -r /tmp/cse234.zip cerebro_ts

export PATH=/opt/py3.7-pyspark2.4-venv/bin/:"$PATH"
export OUTPUT_FILE=/tmp/cerebro.log
export TF_CPP_MIN_LOG_LEVEL=2

exec /opt/spark/bin/spark-submit \
	--num-executors "${EXECUTORS:-4}" \
	--master spark://test-project-master:7077 \
	-c spark.executorEnv.PATH="$PATH" \
	-c spark.executorEnv.TF_CPP_MIN_LOG_LEVEL=2 \
	--py-files /tmp/cse234.zip \
	driver.py \
	--hdfs \
	"$@"
