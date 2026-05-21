import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, size, length
from pyspark.sql.types import IntegerType, DoubleType


spark = SparkSession.builder \
    .appName("Phase3_Stylometric_Features") \
    .getOrCreate()


arabic_prepositions = set([
    "من", "الى", "إلى", "عن", "على", "في", "ب", "ل", "ك",
    "حتى", "منذ", "مذ", "رب", "عدا", "خلا", "حاشا"
])

first_person_words = set([
    "انا", "أنا", "نحن", "لي", "لنا", "عندي", "لدينا",
    "اكتب", "نكتب", "ارى", "أرى", "نرى", "اعتقد", "أعتقد",
    "نعتقد", "اقوم", "أقوم", "نقوم"
])


def avg_word_length(tokens):
    if tokens is None or len(tokens) == 0:
        return 0.0
    total_chars = sum(len(word) for word in tokens)
    return float(total_chars) / len(tokens)


def count_question_marks(text):
    if text is None:
        return 0
    return text.count("?") + text.count("؟")


def count_prepositions(tokens):
    if tokens is None:
        return 0
    return sum(1 for word in tokens if word in arabic_prepositions)


def count_first_person(tokens):
    if tokens is None:
        return 0
    return sum(1 for word in tokens if word in first_person_words)


def entity_diversity(text):
    """
    Approximation for Feature 90:
    Entity Diversity = unique detected entities / total detected entities.

    Since we are not using a full Arabic NER model here, we approximate entities
    as repeated named-like Arabic terms that appear with common academic/entity indicators.
    """
    if text is None:
        return 0.0

    words = text.split()

    entity_candidates = []
    indicators = [
        "جامعه", "جامعة", "وزاره", "وزارة", "مؤسسه", "مؤسسة",
        "شركه", "شركة", "مركز", "معهد", "النظام", "الدوله", "الدولة"
    ]

    for i, word in enumerate(words):
        if word in indicators and i + 1 < len(words):
            entity_candidates.append(word + " " + words[i + 1])

    if len(entity_candidates) == 0:
        return 0.0

    return float(len(set(entity_candidates))) / len(entity_candidates)


avg_word_length_udf = udf(avg_word_length, DoubleType())
question_marks_udf = udf(count_question_marks, IntegerType())
prepositions_udf = udf(count_prepositions, IntegerType())
first_person_udf = udf(count_first_person, IntegerType())
entity_diversity_udf = udf(entity_diversity, DoubleType())


df = spark.read.parquet("data/processed/arabic_ai_detection_parquet")

features_df = df.withColumn("feature_10_avg_word_length", avg_word_length_udf(col("tokens"))) \
    .withColumn("feature_30_question_marks", question_marks_udf(col("text"))) \
    .withColumn("feature_50_prepositions", prepositions_udf(col("tokens"))) \
    .withColumn("feature_70_first_person", first_person_udf(col("tokens"))) \
    .withColumn("feature_90_entity_diversity", entity_diversity_udf(col("clean_text")))

print("===== Feature Engineering Schema =====")
features_df.printSchema()

print("===== Feature Sample =====")
features_df.select(
    "label",
    "generation_method",
    "feature_10_avg_word_length",
    "feature_30_question_marks",
    "feature_50_prepositions",
    "feature_70_first_person",
    "feature_90_entity_diversity"
).show(20, truncate=False)

print("===== Feature Summary by Label =====")
features_df.groupBy("label").avg(
    "feature_10_avg_word_length",
    "feature_30_question_marks",
    "feature_50_prepositions",
    "feature_70_first_person",
    "feature_90_entity_diversity"
).show(truncate=False)

features_df.write.mode("overwrite").parquet("data/processed/phase3_stylometric_features")

print("Stylometric features saved to data/processed/phase3_stylometric_features")

spark.stop()
