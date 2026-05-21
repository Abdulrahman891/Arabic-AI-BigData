import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, lit, array, explode
from pyspark.sql.types import StringType, ArrayType


spark = SparkSession.builder \
    .appName("Phase2_Arabic_Preprocessing") \
    .getOrCreate()


arabic_stopwords = set([
    "في", "من", "على", "إلى", "عن", "أن", "إن", "كان", "كانت", "هو", "هي",
    "هذا", "هذه", "ذلك", "تلك", "ما", "لا", "لم", "لن", "قد", "كل", "كما",
    "مع", "أو", "أي", "بين", "بعد", "قبل", "التي", "الذي", "الذين", "و",
    "ف", "ب", "ل", "ك"
])


def normalize_arabic(text):
    if text is None:
        return ""

    text = str(text)

    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)

    text = re.sub(r"ـ", "", text)

    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_remove_stopwords(text):
    if text is None:
        return []

    words = text.split()
    words = [word for word in words if word not in arabic_stopwords and len(word) > 1]

    return words


normalize_udf = udf(normalize_arabic, StringType())
tokenize_udf = udf(tokenize_remove_stopwords, ArrayType(StringType()))


by_polishing = spark.read.csv(
    "data/raw/by_polishing.csv",
    header=True,
    inferSchema=True,
    multiLine=True,
    quote='"',
    escape='"'
).withColumn("generation_method", lit("by_polishing"))

from_title = spark.read.csv(
    "data/raw/from_title.csv",
    header=True,
    inferSchema=True,
    multiLine=True,
    quote='"',
    escape='"'
).withColumn("generation_method", lit("from_title"))

from_title_and_content = spark.read.csv(
    "data/raw/from_title_and_content.csv",
    header=True,
    inferSchema=True,
    multiLine=True,
    quote='"',
    escape='"'
).withColumn("generation_method", lit("from_title_and_content"))


df = by_polishing.unionByName(from_title, allowMissingColumns=True) \
                 .unionByName(from_title_and_content, allowMissingColumns=True)


human_df = df.select(
    col("original_abstract").alias("text"),
    lit("human").alias("label"),
    col("generation_method")
)

allam_df = df.select(
    col("allam_generated_abstract").alias("text"),
    lit("ai").alias("label"),
    col("generation_method")
)

jais_df = df.select(
    col("jais_generated_abstract").alias("text"),
    lit("ai").alias("label"),
    col("generation_method")
)

llama_df = df.select(
    col("llama_generated_abstract").alias("text"),
    lit("ai").alias("label"),
    col("generation_method")
)

openai_df = df.select(
    col("openai_generated_abstract").alias("text"),
    lit("ai").alias("label"),
    col("generation_method")
)


long_df = human_df.unionByName(allam_df) \
                  .unionByName(jais_df) \
                  .unionByName(llama_df) \
                  .unionByName(openai_df)


processed_df = long_df.withColumn("clean_text", normalize_udf(col("text"))) \
                      .withColumn("tokens", tokenize_udf(col("clean_text")))


print("===== Processed Schema =====")
processed_df.printSchema()

print("===== Total Rows After Reshaping =====")
print(processed_df.count())

print("===== Label Distribution =====")
processed_df.groupBy("label").count().show()

print("===== Sample Processed Rows =====")
processed_df.select("label", "generation_method", "clean_text", "tokens").show(5, truncate=100)
processed_df.write.mode("overwrite").parquet("data/processed/arabic_ai_detection_parquet")

print("Processed data saved successfully as Parquet.")
spark.stop()
