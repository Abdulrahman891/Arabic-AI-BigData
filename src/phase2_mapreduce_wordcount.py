from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode

spark = SparkSession.builder \
    .appName("Phase2_MapReduce_WordCount") \
    .getOrCreate()

df = spark.read.parquet("data/processed/arabic_ai_detection_parquet")

mapped_words = df.select(explode(col("tokens")).alias("word"))

word_counts = mapped_words.groupBy("word").count().orderBy("count", ascending=False)

print("===== MapReduce Word Count Result =====")
word_counts.show(30, truncate=False)

	word_counts.coalesce(1).write.mode("overwrite").csv(
    "reports/figures/mapreduce_word_count",
    header=True
)

print("MapReduce word count saved to reports/figures/mapreduce_word_count")

spark.stop()
