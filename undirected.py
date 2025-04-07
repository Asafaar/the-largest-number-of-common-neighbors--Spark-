import os
import time
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, collect_list, size
from pyspark.sql.types import StructType, StructField, IntegerType
from itertools import combinations
from dotenv import load_dotenv

# === Environment Setup ===
load_dotenv()
os.environ["PYSPARK_PYTHON"] = os.getenv("PYSPARK_PYTHON")
os.environ["PYSPARK_DRIVER_PYTHON"] = os.getenv("PYSPARK_DRIVER_PYTHON")
os.environ["HADOOP_HOME"] = os.getenv("HADOOP_HOME")
os.environ["PATH"] += os.pathsep + os.getenv("PATH_EXTEND")

# === Spark Init ===
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

# Set file location and output parameters

file_path = "data/*.csv"
top_n = 10
t_start = time.time()

# Load CSV with edge list and cast src/dst to integers
edges = spark.read.option("header", "true").csv(file_path)
edges = edges.select(col("src").cast("int"), col("dst").cast("int")) \
             .filter(col("src").isNotNull() & col("dst").isNotNull())

# Since the graph is undirected, we need to make sure edges go both directions
# This way (1,2) and (2,1) are both treated as a connection
edges = edges.union(edges.select(col("dst").alias("src"), col("src").alias("dst")))

# Rearranging the edges: for each node (neighbor), collect who is connected to it
reversed_edges = edges.select(col("dst").alias("neighbor"), col("src").alias("node"))

# Group all nodes pointing to the same neighbor – they'll be checked for commonality
neighbors = reversed_edges.groupBy("neighbor").agg(collect_list("node").alias("nodes"))

# For each group of nodes, generate all possible unordered pairs that share the neighbor
# This helps track which node pairs have which common neighbors
pairs_rdd = neighbors.rdd.flatMap(
    lambda row: [Row(node1=min(a, b), node2=max(a, b)) 
                 for a, b in combinations(row['nodes'], 2)]
    if len(row['nodes']) >= 2 else []
)

# Convert to a Spark DataFrame
pairs_df = spark.createDataFrame(pairs_rdd)

# Count how many times each pair appears – that tells us how many common neighbors they have
common_counts = pairs_df.groupBy("node1", "node2").count() \
                        .withColumnRenamed("count", "common_neighbors")

# Sort by number of common neighbors and pick top N pairs
top_results = common_counts.orderBy(col("common_neighbors").desc()).limit(top_n)
top_results_pd = top_results.toPandas()

t_end = time.time()

# Print final output
print(f"\nTop {top_n} node pairs with most common neighbors (Undirected Graph):")
print(top_results_pd)
print(f"\n🕒 Total time: {t_end - t_start:.2f} seconds")

spark.stop()
