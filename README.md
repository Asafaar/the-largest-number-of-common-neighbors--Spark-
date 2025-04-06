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

## 📂 File Structure

- `edges_1m.csv` – Input graph edge list (columns: `src`, `dst`)
- `benchmark_log.csv` – Automatically generated performance logs
- `top_common_neighbors_YYYYMMDD-HHMMSS.csv` – Output results file

---

## 📄 .env Configuration

Before running, make sure to create a `.env` file in the root directory with the following:

