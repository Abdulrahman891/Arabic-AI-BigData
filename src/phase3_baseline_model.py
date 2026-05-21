from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder \
    .appName("Phase3_Baseline_Model") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

train_df = spark.read.parquet("data/processed/train_data")
validation_df = spark.read.parquet("data/processed/validation_data")

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

train_ready = assembler.transform(train_df).select("features", "label_index")
validation_ready = assembler.transform(validation_df).select("features", "label_index")

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label_index",
    maxIter=20
)

model = lr.fit(train_ready)

predictions = model.transform(validation_ready)

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

print("===== Baseline Model: Logistic Regression =====")
print(f"Validation Accuracy: {accuracy}")
print(f"Validation F1-Score: {f1}")

predictions.groupBy("label_index", "prediction").count().show()

model.write().overwrite().save("models/logistic_regression_baseline")

print("Baseline model saved to models/logistic_regression_baseline")

spark.stop()
