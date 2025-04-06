import os
import time
import csv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, least, greatest
import pyspark.sql.functions as F
from dotenv import load_dotenv

# === Environment Setup ===
load_dotenv()

# Set environment variables
os.environ["PYSPARK_PYTHON"] = os.getenv("PYSPARK_PYTHON")
os.environ["PYSPARK_DRIVER_PYTHON"] = os.getenv("PYSPARK_DRIVER_PYTHON")
os.environ["HADOOP_HOME"] = os.getenv("HADOOP_HOME")
os.environ["PATH"] += os.pathsep + os.getenv("PATH_EXTEND")


# === Event log directory ===
eventlog_dir = "C:/spark-events"
eventlog_uri = f"file:/{eventlog_dir.replace(os.sep, '/')}"

# === Initialize Spark ===
spark = SparkSession.builder \
    .appName("UndirectedCommonNeighborsBenchmark") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", eventlog_uri) \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.sql.inMemoryColumnarStorage.compressed", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.default.parallelism", "100") \
    .config("spark.sql.files.maxPartitionBytes", "134217728") \
    .config("spark.locality.wait", "0s") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# === Parameters ===
file_path = "edges_1m.csv"
top_n = 10
timestamp = time.strftime("%Y%m%d-%H%M%S")

# === Timer total ===
t_total_start = time.time()

# === Step 1: Load CSV ===
t1 = time.time()
edges = spark.read.option("header", "true").csv(file_path)
edges = edges.select(
    least(col("src").cast("int"), col("dst").cast("int")).alias("node1"),
    greatest(col("src").cast("int"), col("dst").cast("int")).alias("node2")
).dropna()
t2 = time.time()

# === Step 2: Reconstruct edges to (dst, src) format ===
# Treat each undirected edge as both directions
edges_reversed = edges.select(
    col("node1").alias("src"),
    col("node2").alias("dst")
).union(
    edges.select(
        col("node2").alias("src"),
        col("node1").alias("dst")
    )
)

edges = edges_reversed.repartition(100).cache()
edges.count()  # Cache trigger
t3 = time.time()

# === Step 3: Join on dst to find shared neighbors ===
src_pairs = edges.alias("a").join(
    edges.alias("b"),
    (col("a.dst") == col("b.dst")) & (col("a.src") < col("b.src"))
).select(
    col("a.src").alias("node1"),
    col("b.src").alias("node2"),
    col("a.dst").alias("common")
).repartition(100, "node1", "node2").cache()
t4 = time.time()

# === Step 4: Group and count ===
common_counts = src_pairs.groupBy("node1", "node2").agg(
    count("common").alias("common_neighbors")
)
t5 = time.time()

# === Step 5: Sort and limit ===
top_results = common_counts.orderBy(col("common_neighbors").desc()).limit(top_n)
top_results_pd = top_results.toPandas()
t6 = time.time()

# === Step 6: Show result
print(f"\nTop {top_n} node pairs with most common neighbors (undirected):")
print(top_results_pd)
t7 = time.time()

# === Timing Summary ===
load_time = t2 - t1
preprocess_time = t3 - t2
join_time = t4 - t3
group_time = t5 - t4
sort_time = t6 - t5
show_time = t7 - t6
total_time = t7 - t_total_start

print("\n⏱️ Detailed Performance Report:")
print(f"✔ Load time        : {load_time:.2f} sec")
print(f"✔ Preprocess time  : {preprocess_time:.2f} sec")
print(f"✔ Join time        : {join_time:.2f} sec")
print(f"✔ Grouping time    : {group_time:.2f} sec")
print(f"✔ Sorting time     : {sort_time:.2f} sec")
print(f"✔ Show time        : {show_time:.2f} sec")
print(f"✔ Total time       : {total_time:.2f} sec")

# === Save Results ===
results_path = f"top_common_neighbors_undirected_{timestamp}.csv"
top_results_pd.to_csv(results_path, index=False)
print(f"\n📁 Results saved to: {results_path}")

# === Save Benchmark Log ===
log_path = "benchmark_log.csv"
log_exists = os.path.exists(log_path)

with open(log_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not log_exists:
        writer.writerow(["timestamp", "file", "type", "top_n", "load_time", "preprocess", "join_time", "group_time", "sort_time", "show_time", "total_time"])
    writer.writerow([timestamp, file_path, "undirected", top_n, load_time, preprocess_time, join_time, group_time, sort_time, show_time, total_time])

print(f"📊 Benchmark log updated: {log_path}")
print(f"📝 Spark event logs saved to: {eventlog_dir}")

# === Cleanup ===
spark.stop()
