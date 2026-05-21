from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.classification import LinearSVC
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

spark = SparkSession.builder \
    .appName("Phase5_PCA_SVM_Boundary") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

os.makedirs("reports/figures", exist_ok=True)

df = spark.read.parquet("data/processed/test_data")

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

data_ready = assembler.transform(df).select("features", "label_index")

pca = PCA(k=2, inputCol="features", outputCol="pca_features")
pca_model = pca.fit(data_ready)
pca_result = pca_model.transform(data_ready)

plot_df = pca_result.select("pca_features", "label_index").toPandas()
plot_df["PC1"] = plot_df["pca_features"].apply(lambda x: float(x[0]))
plot_df["PC2"] = plot_df["pca_features"].apply(lambda x: float(x[1]))
plot_df["Class"] = plot_df["label_index"].map({0: "Human", 1: "AI"})

pc1_low, pc1_high = plot_df["PC1"].quantile([0.01, 0.99])
pc2_low, pc2_high = plot_df["PC2"].quantile([0.01, 0.99])

plot_df_clean = plot_df[
    (plot_df["PC1"] >= pc1_low) & (plot_df["PC1"] <= pc1_high) &
    (plot_df["PC2"] >= pc2_low) & (plot_df["PC2"] <= pc2_high)
].copy()

plot_df_clean = plot_df_clean.sample(
    n=min(1500, len(plot_df_clean)),
    random_state=42
)

X = plot_df_clean[["PC1", "PC2"]].values
y = plot_df_clean["label_index"].values

try:
    from sklearn.svm import LinearSVC as SklearnLinearSVC

    clf = SklearnLinearSVC(max_iter=5000)
    clf.fit(X, y)

    w = clf.coef_[0]
    b = clf.intercept_[0]

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    xx = np.linspace(x_min, x_max, 200)

    yy = -(w[0] * xx + b) / w[1]

    margin = 1 / np.sqrt(np.sum(w ** 2))
    yy_margin_pos = yy + margin
    yy_margin_neg = yy - margin

    draw_boundary = True

except Exception as e:
    print("Could not train sklearn LinearSVC for boundary visualization.")
    print(e)
    draw_boundary = False

plt.figure(figsize=(10, 7), facecolor="#05070A")
ax = plt.gca()
ax.set_facecolor("#05070A")

colors = {
    "AI": "#D6B06A",
    "Human": "#6FA8C9"
}

for label in ["Human", "AI"]:
    subset = plot_df_clean[plot_df_clean["Class"] == label]
    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=label,
        alpha=0.65,
        s=28,
        c=colors[label],
        edgecolors="none"
    )

if draw_boundary:
    plt.plot(xx, yy, color="#F0C36A", linewidth=2.5, label="2D SVM Boundary")
    plt.plot(xx, yy_margin_pos, color="#F0C36A", linewidth=1.2, linestyle="--", alpha=0.8)
    plt.plot(xx, yy_margin_neg, color="#F0C36A", linewidth=1.2, linestyle="--", alpha=0.8)

plt.title("PCA Visualization with Linear SVM Boundary", color="white", fontsize=16)
plt.xlabel("Principal Component 1", color="white")
plt.ylabel("Principal Component 2", color="white")

plt.xticks(color="white")
plt.yticks(color="white")

for spine in ax.spines.values():
    spine.set_color("#D6B06A")

legend = plt.legend(facecolor="#05070A", edgecolor="#D6B06A")
for text in legend.get_texts():
    text.set_color("white")

plt.grid(alpha=0.15)
plt.tight_layout()

output_path = "reports/figures/pca_svm_boundary_visualization.png"
plt.savefig(output_path, dpi=300, facecolor="#05070A")
plt.close()

print(f"PCA SVM boundary visualization saved to {output_path}")

spark.stop()
