import os
import time
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, collect_list, size
from pyspark.sql.types import StructType, StructField, IntegerType
from itertools import combinations
from dotenv import load_dotenv

# === Load environment variables from .env file ===
load_dotenv()
os.environ["PYSPARK_PYTHON"] = os.getenv("PYSPARK_PYTHON")
os.environ["PYSPARK_DRIVER_PYTHON"] = os.getenv("PYSPARK_DRIVER_PYTHON")
os.environ["HADOOP_HOME"] = os.getenv("HADOOP_HOME")
os.environ["PATH"] += os.pathsep + os.getenv("PATH_EXTEND")

# === Initialize SparkSession with performance tuning ===
spark = SparkSession.builder \
    .appName("OptimizedCommonNeighborsBenchmark") \
    .config("spark.eventLog.enabled", "false") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.hadoop.hadoop.native.lib", "false") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.sql.inMemoryColumnarStorage.compressed", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.default.parallelism", "100") \
    .config("spark.sql.files.maxPartitionBytes", "134217728") \
    .config("spark.locality.wait", "0s") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# === Configurable parameters ===
file_path = "data/*.csv"     # Input graph edges file(s)
top_n = 10                   # Number of top pairs to return
MAX_NEIGHBORS = 1000         # Cap to avoid memory explosion per node
t_start = time.time()

# === Load edge list and clean data ===
edges = spark.read.option("header", "true").csv(file_path)
edges = edges.select(col("src").cast("int"), col("dst").cast("int")) \
             .filter(col("src").isNotNull() & col("dst").isNotNull())

# === Flip edges: we want to group by each target node (dst) ===
# This lets us find which source nodes point to the same target
reversed_edges = edges.select(col("dst").alias("neighbor"), col("src").alias("node"))

# === Group by each neighbor and collect the source nodes pointing to it ===
# This is key: each of these nodes share this neighbor in common
neighbors = reversed_edges.groupBy("neighbor").agg(collect_list("node").alias("nodes"))

# === For each neighbor, create all unordered node pairs sharing it ===
# Guard against rows with too few or too many neighbors
pairs_rdd = neighbors.rdd.flatMap(
    lambda row: [
        Row(node1=min(a, b), node2=max(a, b))
        for a, b in combinations(row['nodes'], 2)
    ] if isinstance(row['nodes'], list) and 2 <= len(row['nodes']) <= MAX_NEIGHBORS else []
)

# === Define the schema for the node pairs DataFrame ===
schema = StructType([
    StructField("node1", IntegerType(), False),
    StructField("node2", IntegerType(), False),
])

# === Convert to DataFrame for further processing ===
pairs_df = spark.createDataFrame(pairs_rdd, schema=schema)

# === Count how many times each node pair appeared — number of common neighbors ===
common_counts = pairs_df.groupBy("node1", "node2").count() \
                        .withColumnRenamed("count", "common_neighbors")

# === Select top-N node pairs with the highest number of shared neighbors ===
top_results = common_counts.orderBy(col("common_neighbors").desc()).limit(top_n)
top_results_pd = top_results.toPandas()

t_end = time.time()

# === Final output ===
print(f"\nTop {top_n} node pairs with most common neighbors (Optimized Algorithm):")
print(top_results_pd)
print(f"\n🕒 Total time: {t_end - t_start:.2f} seconds")

spark.stop()
