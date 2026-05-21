import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, current_timestamp, when
from pyspark.sql.types import StringType, ArrayType, IntegerType, DoubleType
from pyspark.ml.feature import CountVectorizerModel, IDFModel, VectorAssembler
from pyspark.ml.classification import LinearSVCModel


spark = SparkSession.builder \
    .appName("Phase4_Streaming_Prediction") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()


arabic_stopwords = set([
    "في", "من", "على", "إلى", "الى", "عن", "أن", "إن", "كان", "كانت", "هو", "هي",
    "هذا", "هذه", "ذلك", "تلك", "ما", "لا", "لم", "لن", "قد", "كل", "كما",
    "مع", "أو", "أي", "بين", "بعد", "قبل", "التي", "الذي", "الذين", "و",
    "ف", "ب", "ل", "ك"
])

arabic_prepositions = set([
    "من", "الى", "إلى", "عن", "على", "في", "ب", "ل", "ك",
    "حتى", "منذ", "مذ", "رب", "عدا", "خلا", "حاشا"
])

first_person_words = set([
    "انا", "أنا", "نحن", "لي", "لنا", "عندي", "لدينا",
    "اكتب", "نكتب", "ارى", "أرى", "نرى", "اعتقد", "أعتقد",
    "نعتقد", "اقوم", "أقوم", "نقوم"
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
    text = re.sub(r"[^\u0600-\u06FF\s؟?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_remove_stopwords(text):
    if text is None:
        return []
    words = text.split()
    return [word for word in words if word not in arabic_stopwords and len(word) > 1]


def avg_word_length(tokens):
    if tokens is None or len(tokens) == 0:
        return 0.0
    return float(sum(len(word) for word in tokens)) / len(tokens)


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


normalize_udf = udf(normalize_arabic, StringType())
tokenize_udf = udf(tokenize_remove_stopwords, ArrayType(StringType()))
avg_word_length_udf = udf(avg_word_length, DoubleType())
question_marks_udf = udf(count_question_marks, IntegerType())
prepositions_udf = udf(count_prepositions, IntegerType())
first_person_udf = udf(count_first_person, IntegerType())
entity_diversity_udf = udf(entity_diversity, DoubleType())


cv_model = CountVectorizerModel.load("models/count_vectorizer_model")
idf_model = IDFModel.load("models/idf_model")
svm_model = LinearSVCModel.load("models/linear_svm_model")


stream_df = spark.readStream \
    .format("text") \
    .load("data/stream_input")


processed_stream = stream_df.withColumnRenamed("value", "text") \
    .withColumn("processing_time", current_timestamp()) \
    .withColumn("clean_text", normalize_udf(col("text"))) \
    .withColumn("tokens", tokenize_udf(col("clean_text"))) \
    .withColumn("feature_10_avg_word_length", avg_word_length_udf(col("tokens"))) \
    .withColumn("feature_30_question_marks", question_marks_udf(col("text"))) \
    .withColumn("feature_50_prepositions", prepositions_udf(col("tokens"))) \
    .withColumn("feature_70_first_person", first_person_udf(col("tokens"))) \
    .withColumn("feature_90_entity_diversity", entity_diversity_udf(col("clean_text")))


cv_stream = cv_model.transform(processed_stream)
tfidf_stream = idf_model.transform(cv_stream)


assembler = VectorAssembler(
    inputCols=[
        "tfidf_features",
        "feature_10_avg_word_length",
        "feature_30_question_marks",
        "feature_50_prepositions",
        "feature_70_first_person",
        "feature_90_entity_diversity"
    ],
    outputCol="features"
)

final_stream = assembler.transform(tfidf_stream)

predictions = svm_model.transform(final_stream)

output = predictions.withColumn(
    "detected_as",
    when(col("prediction") == 1.0, "AI").otherwise("Human")
).select(
    "processing_time",
    "text",
    "prediction",
    "detected_as"
)



query = output.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .option("numRows", 20) \
    .trigger(processingTime="5 seconds") \
    .option("checkpointLocation", "data/stream_checkpoint") \
    .start()

print("Streaming prediction started.")
print("Add .txt files into data/stream_input to get real-time predictions.")

query.awaitTermination()
