from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

spark = SparkSession.builder \
    .appName("Phase1_Initial_Exploration") \
    .getOrCreate()

by_polishing = spark.read.csv(
    "data/raw/by_polishing.csv",
    header=True,
    inferSchema=True,
    multiLine=True,
    quote='"',
    escape='"'
)

from_title = spark.read.csv(
    "data/raw/from_title.csv",
    header=True,
    inferSchema=True,
    multiLine=True,
    quote='"',
    escape='"'
)

from_title_and_content = spark.read.csv(
    "data/raw/from_title_and_content.csv",
    header=True,
    inferSchema=True,
    multiLine=True,
    quote='"',
    escape='"'
)

by_polishing = by_polishing.withColumn(
    "generation_method",
    lit("by_polishing")
)

from_title = from_title.withColumn(
    "generation_method",
    lit("from_title")
)

from_title_and_content = from_title_and_content.withColumn(
    "generation_method",
    lit("from_title_and_content")
)

df = by_polishing.unionByName(from_title, allowMissingColumns=True) \
                 .unionByName(from_title_and_content, allowMissingColumns=True)

print("===== Schema =====")
df.printSchema()

print("===== Total Rows =====")
print(df.count())

print("===== Columns =====")
print(df.columns)

print("===== Generation Method Distribution =====")
df.groupBy("generation_method").count().show()

print("===== Sample Rows =====")
df.show(5, truncate=80)

spark.stop()
