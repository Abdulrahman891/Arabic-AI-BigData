from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("Phase3_Data_Split") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

df = spark.read.parquet("data/processed/phase3_tfidf_features")

df = df.withColumn(
    "label_index",
    (col("label") == "ai").cast("integer")
)

train_df, validation_df, test_df = df.randomSplit([0.70, 0.15, 0.15], seed=42)

train_df.write.mode("overwrite").parquet("data/processed/train_data")
validation_df.write.mode("overwrite").parquet("data/processed/validation_data")
test_df.write.mode("overwrite").parquet("data/processed/test_data")

print("Data splits saved successfully.")
print("Train, validation, and test datasets were created.")

spark.stop()
