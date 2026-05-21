from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, size, countDistinct
from pyspark.ml.feature import NGram

spark = SparkSession.builder \
    .appName("Phase2_EDA") \
    .getOrCreate()

df = spark.read.parquet("data/processed/arabic_ai_detection_parquet")

print("===== Total Rows =====")
print(df.count())

print("===== Label Distribution =====")
df.groupBy("label").count().show()

words_df = df.select("label", explode(col("tokens")).alias("word"))

print("===== Top 20 Words =====")
words_df.groupBy("word").count().orderBy("count", ascending=False).show(20, truncate=False)

ngram = NGram(n=2, inputCol="tokens", outputCol="bigrams")
bigram_df = ngram.transform(df)

bigrams_exploded = bigram_df.select(explode(col("bigrams")).alias("bigram"))

print("===== Top 20 Bigrams =====")
bigrams_exploded.groupBy("bigram").count().orderBy("count", ascending=False).show(20, truncate=False)

total_tokens = words_df.count()
unique_tokens = words_df.select(countDistinct("word")).collect()[0][0]
ttr = unique_tokens / total_tokens

print("===== Vocabulary Richness (TTR) =====")
print(f"Total tokens: {total_tokens}")
print(f"Unique tokens: {unique_tokens}")
print(f"TTR: {ttr}")

spark.stop()
