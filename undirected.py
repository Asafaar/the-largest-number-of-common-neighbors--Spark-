import os
import time
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, collect_list
from pyspark.sql.types import StructType, StructField, IntegerType
from itertools import combinations
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()
os.environ["PYSPARK_PYTHON"] = os.getenv("PYSPARK_PYTHON")
os.environ["PYSPARK_DRIVER_PYTHON"] = os.getenv("PYSPARK_DRIVER_PYTHON")
os.environ["HADOOP_HOME"] = os.getenv("HADOOP_HOME")
os.environ["PATH"] += os.pathsep + os.getenv("PATH_EXTEND")

# === Spark Init ===
spark = SparkSession.builder \
    .appName("UndirectedGraphCommonNeighbors") \
    .config("spark.eventLog.enabled", "false") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# === Parameters ===
file_path = "data/*.csv"
top_n = 10
MAX_NEIGHBORS = 1000
t_start = time.time()

# === Load and double edges to simulate undirected graph ===
edges = spark.read.option("header", "true").csv(file_path)
edges = edges.select(col("src").cast("int"), col("dst").cast("int")) \
             .filter(col("src").isNotNull() & col("dst").isNotNull())

# Add reversed edges to treat the graph as undirected
edges = edges.union(edges.select(col("dst").alias("src"), col("src").alias("dst")))

# === Create reversed view: each neighbor points to nodes that point to it ===
reversed_edges = edges.select(col("dst").alias("neighbor"), col("src").alias("node"))

# === Group neighbors ===
neighbors = reversed_edges.groupBy("neighbor").agg(collect_list("node").alias("nodes"))

# === Generate combinations ===
pairs_rdd = neighbors.rdd.flatMap(
    lambda row: [
        Row(node1=min(a, b), node2=max(a, b))
        for a, b in combinations(row['nodes'], 2)
    ] if isinstance(row['nodes'], list) and 2 <= len(row['nodes']) <= MAX_NEIGHBORS else []
)

# === Schema ===
schema = StructType([
    StructField("node1", IntegerType(), False),
    StructField("node2", IntegerType(), False),
])

# === Create DataFrame from pairs ===
pairs_df = spark.createDataFrame(pairs_rdd, schema=schema)

# === Count common neighbors ===
common_counts = pairs_df.groupBy("node1", "node2").count() \
                        .withColumnRenamed("count", "common_neighbors")

# === Top-N results ===
top_results = common_counts.orderBy(col("common_neighbors").desc()).limit(top_n)
top_results_pd = top_results.toPandas()

t_end = time.time()

# === Output ===
print(f"\nTop {top_n} node pairs with most common neighbors (Undirected Graph):")
print(top_results_pd)
print(f"\n🕒 Total time: {t_end - t_start:.2f} seconds")

spark.stop()
