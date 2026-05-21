from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel, LinearSVCModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import matplotlib.pyplot as plt
import os

spark = SparkSession.builder \
    .appName("Phase5_Model_Comparison_Figures") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

os.makedirs("reports/figures", exist_ok=True)

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

test_ready = assembler.transform(test_df).select("features", "label_index")

models = {
    "Logistic Regression": LogisticRegressionModel.load("models/logistic_regression_baseline"),
    "Random Forest": RandomForestClassificationModel.load("models/random_forest_model"),
    "Linear SVM": LinearSVCModel.load("models/linear_svm_model")
}

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

results = []

for model_name, model in models.items():
    predictions = model.transform(test_ready)

    accuracy = accuracy_evaluator.evaluate(predictions)
    f1 = f1_evaluator.evaluate(predictions)

    cm = predictions.groupBy("label_index", "prediction").count().toPandas()

    tn = cm[(cm["label_index"] == 0) & (cm["prediction"] == 0.0)]["count"].sum()
    fp = cm[(cm["label_index"] == 0) & (cm["prediction"] == 1.0)]["count"].sum()
    fn = cm[(cm["label_index"] == 1) & (cm["prediction"] == 0.0)]["count"].sum()
    tp = cm[(cm["label_index"] == 1) & (cm["prediction"] == 1.0)]["count"].sum()

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "F1-Score": f1,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    })

    matrix = [[tn, fp], [fn, tp]]

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xticks([0, 1], ["Predicted Human", "Predicted AI"])
    plt.yticks([0, 1], ["Actual Human", "Actual AI"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i][j]), ha="center", va="center")

    plt.tight_layout()
    filename = model_name.lower().replace(" ", "_")
    plt.savefig(f"reports/figures/confusion_matrix_{filename}.png", dpi=300)
    plt.close()

results_df = pd.DataFrame(results)
results_df.to_csv("reports/figures/model_comparison_results.csv", index=False)

print("===== Model Comparison Results =====")
print(results_df)

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("reports/figures/model_accuracy_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["F1-Score"])
plt.title("Model F1-Score Comparison")
plt.ylabel("F1-Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("reports/figures/model_f1_comparison.png", dpi=300)
plt.close()

print("Figures saved in reports/figures")

spark.stop()
