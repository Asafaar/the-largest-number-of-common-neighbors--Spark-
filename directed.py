import os
import time
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, collect_list
from pyspark.sql.types import StructType, StructField, IntegerType
from itertools import combinations
from dotenv import load_dotenv

# Load environment variables from .env file for local config
load_dotenv()
os.environ["PYSPARK_PYTHON"] = os.getenv("PYSPARK_PYTHON")
os.environ["PYSPARK_DRIVER_PYTHON"] = os.getenv("PYSPARK_DRIVER_PYTHON")
os.environ["HADOOP_HOME"] = os.getenv("HADOOP_HOME")
os.environ["PATH"] += os.pathsep + os.getenv("PATH_EXTEND")

# Start Spark session with custom memory and shuffle tuning
spark = SparkSession.builder \
    .appName("DirectedGraphCommonNeighbors") \
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

# Path to input file(s)
file_path = "data/*.csv"
top_n = 10
t_start = time.time()

# Load edges from file and ensure types are consistent
edges = spark.read.option("header", "true").csv(file_path)
edges = edges.select(col("src").cast("int"), col("dst").cast("int")) \
             .filter(col("src").isNotNull() & col("dst").isNotNull())

# No reversal: we keep edges as-is since direction matters in directed graphs

# Each dst (target) becomes a 'common neighbor' candidate
# We want to find out which source nodes point to the same target
reversed_edges = edges.select(col("dst").alias("neighbor"), col("src").alias("node"))

# Group all nodes that share the same neighbor (i.e., pointing to same dst)
neighbors = reversed_edges.groupBy("neighbor").agg(collect_list("node").alias("nodes"))

# From each group, generate all unordered node pairs – these are potential common neighbor pairs
pairs_rdd = neighbors.rdd.flatMap(
    lambda row: [Row(node1=min(a, b), node2=max(a, b))
                 for a, b in combinations(row['nodes'], 2)]
    if len(row['nodes']) >= 2 else []
)

# Convert the pairs into a structured DataFrame
schema = StructType([
    StructField("node1", IntegerType(), False),
    StructField("node2", IntegerType(), False),
])
pairs_df = spark.createDataFrame(pairs_rdd, schema=schema)

# Count how many times each node pair appeared (i.e., how many common dsts they share)
common_counts = pairs_df.groupBy("node1", "node2").count() \
                        .withColumnRenamed("count", "common_neighbors")

# Return top-N most connected node pairs by shared neighbors
top_results = common_counts.orderBy(col("common_neighbors").desc()).limit(top_n)
top_results_pd = top_results.toPandas()

t_end = time.time()

# Print the result
print(f"\nTop {top_n} node pairs with most common neighbors (Directed Graph):")
print(top_results_pd)
print(f"\n🕒 Total time: {t_end - t_start:.2f} seconds")

spark.stop()
