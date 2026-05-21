from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LinearSVCModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder \
    .appName("Phase4_Final_Evaluation") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

test_df = spark.read.parquet("data/processed/test_data")

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

test_ready = assembler.transform(test_df).select(
    "features",
    "label_index"
)

model = LinearSVCModel.load("models/linear_svm_model")

predictions = model.transform(test_ready)

accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="label_index",
    predictionCol="prediction",
    metricName="accuracy"
)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label_index",
    predictionCol="prediction",
    metricName="f1"
)

accuracy = accuracy_evaluator.evaluate(predictions)
f1 = f1_evaluator.evaluate(predictions)

print("===== Final Test Evaluation =====")
print(f"Test Accuracy: {accuracy}")
print(f"Test F1-Score: {f1}")

print("===== Confusion Matrix =====")
predictions.groupBy("label_index", "prediction").count().show()

spark.stop()
