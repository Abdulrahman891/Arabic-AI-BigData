from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("Phase3_TFIDF") \
    .getOrCreate()

df = spark.read.parquet("data/processed/phase3_stylometric_features")

print("===== Original Dataset =====")
print(df.count())

cv = CountVectorizer(
    inputCol="tokens",
    outputCol="raw_features",
    vocabSize=5000,
    minDF=5
)

cv_model = cv.fit(df)

cv_df = cv_model.transform(df)

idf = IDF(
    inputCol="raw_features",
    outputCol="tfidf_features"
)

idf_model = idf.fit(cv_df)

tfidf_df = idf_model.transform(cv_df)

print("===== TF-IDF Schema =====")
tfidf_df.printSchema()

print("===== TF-IDF Sample =====")
tfidf_df.select(
    "label",
    "tokens",
    "tfidf_features"
).show(5, truncate=False)

tfidf_df.write.mode("overwrite").parquet(
    "data/processed/phase3_tfidf_features"
)

print("TF-IDF dataset saved successfully.")
cv_model.write().overwrite().save("models/count_vectorizer_model")
idf_model.write().overwrite().save("models/idf_model")

print("CountVectorizer and IDF models saved successfully.")
spark.stop()

