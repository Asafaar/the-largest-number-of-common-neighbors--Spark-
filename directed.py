import os
import time
import csv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, broadcast
from pyspark.sql import Window
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
# eventlog_dir = "C:/spark-events"
# eventlog_uri = f"file:/{eventlog_dir.replace(os.sep, '/')}"
   # .config("spark.eventLog.dir", eventlog_uri) \
# === Initialize Spark with optimized configuration ===
spark = SparkSession.builder \
    .appName("OptimizedCommonNeighborsBenchmark") \
    .config("spark.eventLog.enabled", "true") \
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
data_dir = "data"

top_n = 10

# === Timer total ===
t_total_start = time.time()

# === Step 1: Load CSV efficiently ===
t1 = time.time()
edges = spark.read.option("header", "true").csv(data_dir)
edges = edges.select(col("src").cast("int"), col("dst").cast("int"))
edges = edges.filter(col("src").isNotNull() & col("dst").isNotNull())

# Cache the edges DataFrame since we'll reuse it
edges = edges.repartition(100).cache()
edges.count()  # Force cache population
t2 = time.time()

# === Step 2: Join to find common neighbors - optimized with broadcast for smaller dataset ===
# Create a directed edge list and get distinct pairs
t3 = time.time()

# Use broadcast join if appropriate for your dataset size
src_pairs = edges.alias("a").join(
    edges.alias("b"),
    (col("a.dst") == col("b.dst")) & (col("a.src") < col("b.src"))
).select(
    col("a.src").alias("node1"),
    col("b.src").alias("node2"),
    col("a.dst").alias("common")  # Include common neighbor for debugging/validation
)

# We can cache at this stage to speed up the next operations
src_pairs = src_pairs.repartition(100, "node1", "node2").cache()
t4 = time.time()

# === Step 3: Group and Count - more efficiently ===
common_counts = src_pairs.groupBy("node1", "node2").agg(count("common").alias("common_neighbors"))


t5 = time.time()

# === Step 4: Sort and Limit ===
top_results = common_counts.orderBy(col("common_neighbors").desc()).limit(top_n)
top_results_pd = top_results.toPandas()
t6 = time.time()

# === Step 5: Show Results ===
print(f"\nTop {top_n} node pairs with most common neighbors:")
print(top_results_pd)
t7 = time.time()

# === Timings ===
load_time = t2 - t1
preprocessing_time = t3 - t2
join_time = t4 - t3
group_time = t5 - t4
sort_time = t6 - t5
show_time = t7 - t6
total_time = t7 - t_total_start

print("\n⏱️ Detailed Performance Report:")
print(f"✔ Load time        : {load_time:.2f} sec")
print(f"✔ Preprocessing    : {preprocessing_time:.2f} sec")
print(f"✔ Join time        : {join_time:.2f} sec")
print(f"✔ Grouping time    : {group_time:.2f} sec")
print(f"✔ Sorting time     : {sort_time:.2f} sec")
print(f"✔ Show time        : {show_time:.2f} sec")
print(f"✔ Total time       : {total_time:.2f} sec")

# === Step 6: Save top results
timestamp = time.strftime("%Y%m%d-%H%M%S")
results_path = f"top_common_neighbors_{timestamp}.csv"
top_results_pd.to_csv(results_path, index=False)
print(f"\n📁 Results saved to: {results_path}")

# === Step 7: Save benchmark summary to CSV log
log_path = "benchmark_log.csv"
log_exists = os.path.exists(log_path)

with open(log_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not log_exists:
        writer.writerow(["timestamp", "file", "top_n", "load_time", "preprocessing", "join_time", "group_time", "sort_time", "show_time", "total_time"])
    writer.writerow([timestamp, file_path, top_n, load_time, preprocessing_time, join_time, group_time, sort_time, show_time, total_time])

print(f"📊 Benchmark log updated: {log_path}")
# print(f"📝 Spark event logs saved to: {eventlog_dir}")

# Cleanup
spark.stop()
