# 🔗 Optimized Common Neighbors Benchmark using PySpark

This project efficiently computes the **Top-N node pairs** in a graph that share the **highest number of common neighbors**. It leverages the power of **Apache Spark** to process large-scale edge lists with **hundreds of thousands to millions of edges** in a distributed and optimized way.

---

## 🚀 Features and Optimizations

- ✅ **Handles large graphs** (1M+ edges) with strong performance
- ✅ **PySpark DataFrame API** used throughout for distributed execution
- ✅ **Optimized hash aggregations**
- ✅ **Repartitioning + caching** to reduce recomputation
- ✅ **Memory configuration tuning** for stability and speed
- ✅ **Performance logging** to compare versions over time

---

## 🧪 Benchmarks

| Dataset Size      | Runtime (approx.) |
|-------------------|-------------------|
| 500,000 edges     | ~40 seconds       |
| 1,000,000 edges   | ~78 seconds       |

> 🧠 **Note:** These benchmarks were obtained on a local Windows machine with fixed hardware. Performance can be improved by providing better memory and disk resources or scaling to a Spark cluster.

---

## 🧮 Time and Space Complexity Analysis

This implementation uses an **optimized graph-based approach** rather than a brute-force join method. Here's why it's significantly more efficient:

### ⏱️ Time Complexity
- For each node (considered as a common neighbor), we generate all unordered pairs from its neighbors.
- For a node with degree d, we generate O(d^2) pairs.
- Over all n nodes, this results in:

**O(n * d^2) ≈ O(E^2 / n)**

Where E is the total number of edges.

### 🧠 Space Complexity
- We temporarily store all generated node pairs (per common neighbor) before aggregation.
- This again is **O(n * d^2)** in memory, which is significantly better than the potential **O(E^2)** of a join-based approach.

### 🚀 Why This is the Most Efficient Path
- No expensive JOIN operations → avoids shuffle-heavy operations.
- Linear in the number of nodes (n), quadratic only in degree (d) which is low in sparse graphs.
- Scales excellently in distributed environments (Spark partitions the graph efficiently).

---

## 📂 File Structure

- `edges_1m.csv` – Input graph edge list (columns: `src`, `dst`)
- `benchmark_log.csv` – Automatically generated performance logs
- `top_common_neighbors_YYYYMMDD-HHMMSS.csv` – Output results file

---

## 📄 .env Configuration

Before running, make sure to create a `.env` file in the root directory with the following:
