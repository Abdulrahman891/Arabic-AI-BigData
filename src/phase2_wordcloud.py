from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
import os

spark = SparkSession.builder \
    .appName("Phase2_WordCloud") \
    .getOrCreate()

df = spark.read.parquet("data/processed/arabic_ai_detection_parquet")

words_df = df.select(explode(col("tokens")).alias("word"))

top_words = words_df.groupBy("word") \
    .count() \
    .orderBy("count", ascending=False) \
    .limit(300) \
    .collect()

word_freq = {}

for row in top_words:
    reshaped_word = arabic_reshaper.reshape(row["word"])
    bidi_word = get_display(reshaped_word)
    word_freq[bidi_word] = row["count"]

os.makedirs("reports/figures", exist_ok=True)

font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

wordcloud = WordCloud(
    font_path=font_path,
    width=1200,
    height=800,
    background_color="white",
    max_words=200
).generate_from_frequencies(word_freq)

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()

output_path = "reports/figures/arabic_wordcloud.png"
plt.savefig(output_path, dpi=300)
print(f"Word cloud saved to: {output_path}")

spark.stop()
