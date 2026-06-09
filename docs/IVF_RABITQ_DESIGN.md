# IVF+RaBitQ Index: Design & Implementation

## Table of Contents

1. [Overview](#overview)
2. [RaBitQ Quantization](#rabitq-quantization)
3. [IVF Index Architecture](#ivf-index-architecture)
4. [Memory-Efficient Streaming Build](#memory-efficient-streaming-build)
5. [Search: FastScan SIMD Batch Distance Computation](#search-fastscan-simd-batch-distance-computation)
6. [Incremental Insert](#incremental-insert)
7. [Object Storage Persistence (V4)](#object-storage-persistence-v4)
8. [Delta Segment Design](#delta-segment-design)
9. [Test Results](#test-results)

---

## Overview

IVF+RaBitQ is a high-performance approximate nearest neighbor search index combining
**Inverted File (IVF)** clustering with **RaBitQ quantization**.  Vectors are partitioned
into clusters via K-Means; each cluster is independently quantized with RaBitQ for compact
storage and SIMD-accelerated batch distance computation.

```
┌──────────────────────────────────────────────────────────────────┐
│                     IVF + RaBitQ Index                           │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │ Cluster 0 │  │ Cluster 1 │  │   ...    │  │ Cluster 4095 │    │
│  │           │  │           │  │          │  │              │    │
│  │ centroid  │  │ centroid  │  │          │  │ centroid     │    │
│  │   +       │  │   +       │  │          │  │   +          │    │
│  │ RaBitQ    │  │ RaBitQ    │  │          │  │ RaBitQ       │    │
│  │ codes     │  │ codes     │  │          │  │ codes        │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘    │
│                                                                  │
│  Storage per vector: ~868 bytes (7-bit)                          │
│    · binary code:    padded_dim × 1/8  ≈ 120 B                  │
│    · ex_code:        padded_dim × 6/8  ≈ 720 B                  │
│    · params:         7 × f32           ≈  28 B                  │
└──────────────────────────────────────────────────────────────────┘
```

| Metric | Value |
|--------|-------|
| GIST-1M, 1M vectors, 4096 clusters, 7-bit | Build: 74s, Peak RSS: 1.88 GB |
| Index size (memory) | ~952 MB |
| Search nprobe=96, Recall@100 | 89.0% |
| Search nprobe=128, Recall@100 | 92.2% |
| Search QPS (nprobe=64, topk=10) | ~1,100 |

---

## RaBitQ Quantization

### Algorithm

RaBitQ (Rotated Adaptive Binary Ternary Quantization) encodes each vector as:

1. **Binary code** (1 bit/dim): sign of each dimension after rotation
2. **Extended code** (ex_bits/dim): refines the residual for higher accuracy
3. **Reconstruction parameters**: δ (delta), vl, f_add, f_rescale, f_error, f_add_ex, f_rescale_ex

```
Quantization:
  v_rotated = rotate(v)           // orthogonal random rotation (FHT)
  binary_code[d] = sign(v_rotated[d])  // 1 bit per dimension
  
  residual[d] = v_rotated[d] - binary_code[d] × delta + vl
  ex_code[d]   = quantize(residual[d], ex_bits)  // multi-bit refinement
  
  f_add, f_rescale = regression_params(binary_code, v_rotated)
  f_add_ex, f_rescale_ex = regression_params(ex_code, residual)
```

### Distance Estimation

For L2 distance between query `q` and a database vector:

```
est_dist = f_add + g_add + f_rescale × binary_term
           + f_add_ex + g_add + f_rescale_ex × (binary_scale × binary_dot + ex_dot + kbx_sum_q)
```

where:
- `g_add` = centroid-query distance (precomputed per cluster)
- `binary_term` = dot(binary_code, rotated_query) + k1x_sum_q
- `ex_dot` = dot(ex_code, rotated_query) via specialized SIMD inner product

### Configuration

| Parameter | Description | Typical value |
|-----------|-------------|---------------|
| `total_bits` | Total bits per dimension | 7 (1 binary + 6 extended) |
| `ex_bits` | Extended bits per dimension | `total_bits - 1` |
| `padded_dim` | Dimension after rotation padding | next multiple of 64 ≥ dim |

The **FHT (Fast Hadamard Transform)** rotation evenly distributes variance across
dimensions, making 1-bit quantization more effective.  The `faster_config` mode uses
a precomputed scaling factor for 100-500× faster quantization with <1% accuracy loss.

---

## IVF Index Architecture

### Training

1. **Reservoir sampling**: stream the full dataset once, reservoir-sample `nlist × 64`
   vectors for K-Means training.

2. **K-Means**: Lloyd's algorithm on the rotated reservoir sample.  GEMM via
   [`faer::matmul`](https://faer.veganb.tw) with `Par::rayon(0)` (multi-core).
   15 iterations, 1 restart, `max_points_per_centroid=64`.

3. **Streaming rotation + quantization**: stream the dataset a second time, rotate
   each batch in parallel (`rayon`), assign to nearest centroid via GEMM (faer),
   and quantize with RaBitQ.

4. **Cluster construction**: pack quantized vectors into FastScan batch layout
   (32 vectors per batch) in contiguous memory.

### Memory Layout (`ClusterData`)

```
ClusterData (per cluster)
├── centroid: Vec<f32>                         // padded_dim × 4 B
├── ids: Vec<usize>                            // n × 8 B
├── batch_data: Vec<u8>                        // contiguous FastScan layout
│   └── [Batch 0][Batch 1]...[Batch N]
│       └── per batch (stride = pd×4 + 32×12 B):
│           ├── packed_binary_codes: pd × 32 / 8 B
│           ├── f_add:    32 × 4 B
│           ├── f_rescale: 32 × 4 B
│           └── f_error:  32 × 4 B
├── ex_codes_packed: Vec<Vec<u8>>              // per-vector packed ex codes
├── f_add_ex, f_rescale_ex: Vec<f32>           // per-vector ex params
├── delta, vl: Vec<f32>                        // reconstruction params
├── pending_ids, pending_vectors               // insert buffer (≤31 vectors)
└── num_vectors, padded_dim, ex_bits           // metadata
```

---

## Memory-Efficient Streaming Build

### Problem

The naive build path loads all vectors into memory (~4 GB for 1M × 960), rotates
them all at once (another ~4 GB), and quantizes in parallel (~1 GB temporary).
Peak RSS reaches **~16 GB** for GIST-1M with 4096 clusters.

### Streaming Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streaming Build Pipeline                  │
│                                                             │
│  Pass 1: BatchReader → reservoir sample (262K vectors)     │
│           ↓                                                 │
│  Pass 2: BatchReader → sub-chunk → rotate (parallel)       │
│                         ↓                                   │
│                      GEMM assign (faer, multi-core)         │
│                         ↓                                   │
│                      quantise (rayon, parallel)             │
│                         ↓                                   │
│                      append → next sub-chunk                │
│                                                             │
│  Each sub-chunk: 20K vectors bounded                        │
│  Peak memory: ~1.88 GB (reservoir + single GEMM buffer     │
│                          + clusters under construction)     │
└─────────────────────────────────────────────────────────────┘
```

### Key Optimizations

| Optimization | Before | After | Savings |
|-------------|--------|-------|---------|
| **BatchReader** — streaming file reads | All vectors in memory (~4 GB) | One batch at a time (~366 MB), freed immediately | ~4 GB |
| **Sub-chunk processing** (20K vectors) | GEMM dot_products: 100K × 4096 × 4 = 1.56 GB | 20K × 4096 × 4 = 312 MB | ~1.2 GB |
| **Pre-allocated parallel rotation** | `flat_map` creates 1.26M temporary Vecs | `par_chunks_mut` into pre-allocated buffer | ~1 GB + allocation overhead |
| **Single GEMM buffer in k-means** | rayon fold: 8 threads × 512 MB = 4 GB | Sequential loop: 1 × 512 MB, faer internal multi-core | ~3.5 GB |
| **faer replaces matrixmultiply** | matrixmultiply threading: 2-layer parallel conflict | faer `Par::rayon(0)`: single-layer parallelism | No memory impact; 2× faster |

### GEMM Backend: faer

The [`faer`](https://faer.veganb.tw) crate provides hand-optimized SIMD micro-kernels
with native rayon parallelism.  Replacing `matrixmultiply` eliminates the two-layer
parallelism conflict (outer rayon fold × inner threading) and delivers **2–2.6× faster**
GEMM with lower memory:

```
Old (matrixmultiply + rayon fold):
  rayon fold(8 chunks) × sgemm(threading)
  → core contention + thread-tree overhead
  → 114.7s k-means, 5.6 GB peak

New (faer, sequential chunks):
  for chunk in 0..8: faer::matmul(Par::rayon(0))
  → full 16 cores per matmul, no contention
  → 43.8s k-means, 1.88 GB peak
```

### Build Phase Timing (GIST-1M, 4096 clusters, 16 cores)

```
Phase 1 (streaming reservoir):              6.3s    9%
Phase 2:
  rotate reservoir:                         0.3s    0%
  k-means (15 iter, faer):                 43.8s   59%
  stream rotate (parallel):                 1.1s    1%
  stream GEMM + argmin (faer):             15.5s   21%
  stream quantise (rayon):                  1.0s    1%
  stream append:                            0.5s    1%
  flush_pending:                            0.1s    0%
  save to disk:                             ~6s    8%
Total:                                      74s
```

---

## Search: FastScan SIMD Batch Distance Computation

Search proceeds in two phases:

### Phase 1: Cluster Selection

1. Rotate query via FHT: `q_rotated = rotate(query)`
2. Compute L2 distance from `q_rotated` to every cluster centroid
3. Partial-sort to select top `nprobe` clusters

### Phase 2: FastScan Batch Search

For each selected cluster, process vectors in batches of 32:

```
For each batch of 32 vectors:
  1. Build LUT from rotated query:
     · 4 dimensions → 16 LUT entries (2^4 bit patterns)
     · Quantized to i8 for SIMD accumulation

  2. SIMD accumulate (AVX2):
     · For each 4-dim codebook, index LUT by 4-bit binary code
     · Accumulate in u16 registers (32 vectors × 1 u16 each)

  3. Estimate distances:
     · ip_x0_qr = delta_lut × accu + sum_vl_lut
     · est_dist = f_add + g_add + f_rescale × (ip_x0_qr + k1x_sum_q)

  4. Lower-bound pruning:
     · lower_bound = f(est_dist, centroid_norm, query_norm, dot_centroid_query)
     · Skip if lower_bound ≥ current k-th best distance

  5. Ex-code refinement (for survivors):
     · Unpack ex_code on-demand (C++-style lazy unpacking)
     · Compute ex_dot via specialized SIMD inner product
     · refine_dist = f_add_ex + g_add + f_rescale_ex × total_term

  6. Update distance heap (top-k)
```

### High-Accuracy Mode

For high-dimensional vectors (`padded_dim > 2048`), the LUT uses a 16-bit split
representation (low8/high8) to prevent overflow in u16 accumulators.

---

## Incremental Insert

### Design Goals

1. **O(1) amortized** per-vector insert cost
2. **Minimal memory** during insert phase (centroids only, ~18 MB for 4096 clusters)
3. **Streaming batch** compatible: insert arbitrary-sized batches, flush whenever ready
4. **Object storage friendly**: delta files, no modification of existing data

### Pending Buffer

Vectors are not immediately packed into the FastScan batch layout.  Instead, they
accumulate in a **pending buffer**:

```rust
struct ClusterData {
    // ... existing batch_data for committed vectors ...
    pending_ids: Vec<usize>,         // ≤31 IDs
    pending_vectors: Vec<QuantizedVector>,  // ≤31 pre-quantized vectors
}
```

```
insert_batch(flat_batch):
  1. Rotate batch: [n×dim] → [n×padded_dim] (parallel, pre-allocated)
  2. GEMM centroid assignment (faer::matmul, multi-core)
  3. Parallel quantize: rayon over n vectors
  4. append_vector(cid, qvec):
       pending_ids.push(id)
       pending_vectors.push(qvec)
       if pending_ids.len() >= 32:
           flush_pending()  // pack into batch_data

flush_pending():
  · Drain 32 vectors, pack into 1 batch stride
  · If < 32 remain, pad with zeros and pack
  · Update num_vectors, ids, batch_data, ex_codes, params
```

### Insert Performance

| Dataset | Batch size | Throughput |
|---------|-----------|------------|
| GIST-960, 50K vectors | 50,000 | ~25,000 vec/s |
| GIST-960, 10K vectors | 10,000 | ~28,000 vec/s |

### Search During Insert

Pending vectors (≤31 per cluster) are scanned one-by-one alongside FastScan batch search.
This adds negligible overhead since pending vectors are rare and per-cluster.

### Builder API

```rust
// Lazy load: centroids only (~18 MB), ready for insert
let mut builder = IvfRabitqBuilder::load(store, dim, nlist, bits, ...).await?;

// Insert batches (no IO)
builder.insert_batch(flat_batch_1)?;
builder.insert_batch(flat_batch_2)?;

// Flush: write delta segments, update manifest
let index = builder.flush(store).await?;
```

---

## Object Storage Persistence (V4)

### Design Principles

1. **Immutable files**: all segment files are write-once, never modified
2. **Manifest + Segments**: single manifest.bin describing all segments
3. **Idempotent**: manifest can be read to reconstruct full index state
4. **Cloud-native**: uses `object_store` crate (S3, GCS, Azure Blob, local FS)

### File Layout

```
{prefix}/
├── manifest.bin                 ← V2 manifest (cluster metadata + segment list)
├── cluster_0000_v0000.seg       ← Base segment (initial build, immutable)
├── cluster_0000_v0001.seg       ← Delta segment (1st incremental flush, immutable)
├── cluster_0000_v0002.seg       ← Delta segment (2nd incremental flush, immutable)
├── cluster_0001_v0000.seg
├── cluster_0001_v0001.seg
├── ...
```

### Manifest Format

```
Manifest V2 binary layout:
  magic[4] = "RBQ3"
  version[4] = 2
  dim[4], padded_dim[4]
  metric[1], rotator_type[1]
  ex_bits[1], total_bits[1]
  _placeholder[8]
  rotator_data_len[8], rotator_data[...]
  cluster_count[4]
  for each cluster:
      cluster_id[4]
      segment_count[4]
      for each segment:
          segment_version[4]   // 0=base, 1+=delta
          num_vectors[4]
          file_size[8]
          filename_len[4], filename[...]
  checksum[4]
```

Backward compatible: V1 manifests (single segment per cluster) are auto-upgraded on read.

### Segment Format

```
Segment binary layout:
  magic[4] = "SEG1"
  cluster_id[4], segment_version[4]
  padded_dim[4], ex_bits[1], num_vectors[4]
  centroid[padded_dim × 4]
  ids[num_vectors × 8]
  batch_data_len[8], batch_data[...]
  ex_codes_count[4]
  for each ex_code: len[8], data[...]
  f_add_ex[num_vectors × 4], f_rescale_ex[num_vectors × 4]
  delta[num_vectors × 4], vl[num_vectors × 4]
  checksum[4]
```

### Range-Read Centroid Extraction

During lazy load for insert, only the header + centroid prefix is downloaded:

```
Segment layout prefix (bytes 0..21+pd×4):
  [0..4)   magic
  [4..8)   cluster_id
  [8..12)  segment_version
  [12..16) padded_dim
  [16]     ex_bits
  [17..21) num_vectors
  [21..21+pd×4) centroid  ← range-read stops here
```

`store.get_range(0..21 + padded_dim × 4)` downloads ~3.8 KB per cluster instead of the
entire segment (hundreds of KB).  For 4096 clusters: **15 MB vs 1.2 GB**.

---

## Delta Segment Design

### Motivation

Object stores (S3, GCS, Azure Blob) have **immutable objects**.  You cannot append to or
modify an existing object — you can only PUT new objects and DELETE old ones.

The naive approach (read old segment → merge with new data → write merged segment →
delete old segment) has several problems:

1. **Read amplification**: must download the full old segment for every dirty cluster
2. **Write amplification**: rewrite all old data even if only adding a few new vectors
3. **DELETE cost**: S3 DELETE requests are not free
4. **Concurrency risk**: delete-then-write is not atomic

### Solution: Delta Segments

Each incremental flush writes a **delta segment** containing ONLY the new vectors.
Existing segments are never read, modified, or deleted.

```
Round 0 (initial build):
  cluster_0042_v0000.seg  ← 781 vectors (base)

Round 1 (insert 10K, flush):
  cluster_0042_v0001.seg  ← 82 new vectors (delta)

Round 2 (insert 10K, flush):
  cluster_0042_v0002.seg  ← 79 new vectors (delta)
```

**Search**: `load_from_v4()` reads ALL segments (base + all deltas) for each cluster
and merges them in memory.

**Insert**: `IvfRabitqBuilder::load()` reads only centroids via range-reads of the
base segment.  All inserts are in-memory.

**Flush**: `IvfRabitqBuilder::flush()` writes delta segments for dirty clusters and
atomically PUTs the updated manifest.  No reads, no deletions.

### IO Comparison

| Operation | Merge-based Flush | Delta-based Flush |
|-----------|-------------------|-------------------|
| Reads (flush) | Old segment data (~85 MB) | **0** |
| Writes (flush) | Merged segments + manifest (~86 MB) | Delta segments only (~9 MB for 10K vectors) |
| Deletes | 121 old segments | **0** |
| Load for insert | Full segments (~85 MB) | Centroids only (~488 KB) |

### Future: Compaction

When a cluster accumulates too many delta segments (e.g., >10), a background compaction
job reads all segments, merges them, writes a new base segment, and deletes the old ones.
This is a separate, non-blocking operation.

---

## Test Results

### Environment

- CPU: Intel Xeon with AVX2, 16 physical cores
- Dataset: GIST-1M (960-dimensional, 1M vectors, 1K queries)
- Object store: MinIO @ localhost:9000 (S3-compatible) for IO tests; local FS for timing
- Build: streaming V4, `nlist=4096`, `bits=7`, `faster_config`, `batch_size=100000`

### Build Performance (1M vectors, 4096 clusters, local FS)

| Phase | Time | % |
|-------|------|-----|
| Phase 1: reservoir sampling (streaming read) | 6.3s | 9% |
| Rotate reservoir | 0.3s | 0% |
| K-Means (15 iterations, faer) | 43.8s | 59% |
| Stream rotate (parallel, rayon) | 1.1s | 1% |
| Stream GEMM + argmin (faer + rayon) | 15.5s | 21% |
| Stream quantise (rayon) | 1.0s | 1% |
| Stream append | 0.5s | 1% |
| Flush pending | 0.1s | 0% |
| Save to disk | ~6s | 8% |
| **Total** | **74s** | |

| Metric | Value |
|--------|-------|
| Peak RSS | **1.88 GB** |
| Index memory (steady state) | 952 MB |
| Index on disk | ~867 MB (4096 segments + manifest) |

### Build Memory Evolution

| Implementation | Peak RSS | Build Time | K-Means Time |
|---------------|----------|------------|-------------|
| V3: full in-memory `train()` | ~16 GB | 89s | — |
| V4 streaming + matrixmultiply (rayon fold) | 5.6 GB | 162s | 114.7s |
| V4 streaming + matrixmultiply (sequential k-means) | 1.88 GB | 193s | ~170s |
| V4 streaming + pre-alloc rotation + sub-chunk | 2.1 GB | 163s | ~100s |
| **V4 streaming + faer** | **1.88 GB** | **74s** | **43.8s** |

### Search: Recall@10 (1M vectors, 4096 clusters, local FS)

| nprobe | QPS | Recall@10 |
|--------|-----|-----------|
| 1 | 1,823 | 0.1605 |
| 4 | 1,707 | 0.3701 |
| 8 | 1,656 | 0.5049 |
| 16 | 1,392 | 0.6549 |
| 32 | 1,375 | 0.7878 |
| 64 | 1,126 | 0.8835 |
| 96 | 995 | 0.9208 |
| 128 | 845 | 0.9393 |
| 256 | 589 | 0.9613 |
| 512 | 372 | 0.9683 |
| 1024 | 232 | 0.9685 |

### Search: Recall@100 (1M vectors, 4096 clusters, local FS)

| nprobe | QPS | Recall@100 |
|--------|-----|------------|
| 1 | 1,464 | 0.1083 |
| 4 | 1,235 | 0.2749 |
| 8 | 1,145 | 0.4059 |
| 16 | 1,061 | 0.5497 |
| 32 | 951 | 0.6997 |
| 64 | 761 | 0.8304 |
| 96 | 689 | 0.8898 |
| 128 | 638 | 0.9216 |
| 256 | 474 | 0.9667 |
| 512 | 328 | 0.9788 |
| 1024 | 208 | 0.9804 |

### Recall Validation

| Method | nprobe | Recall@100 | Notes |
|--------|--------|-----------|-------|
| **This implementation (RaBitQ 7-bit)** | 64 | 83.0% | faer GEMM |
| **This implementation (RaBitQ 7-bit)** | 128 | 92.2% | faer GEMM |
| **This implementation (RaBitQ 7-bit)** | 256 | 96.7% | faer GEMM |
| FAISS IVF+PQ (8-bit, 4096) | 64 | ~85–90% | Reference baseline |
| FAISS IVF+PQ (8-bit, 4096) | 128 | ~92–95% | Reference baseline |

Recall is competitive with FAISS IVF+PQ at equivalent nprobe values.

### Incremental Insert on S3 (1M base + 10K insert, 4096 clusters)

| Metric | Value |
|--------|-------|
| Load (lazy) reads | **15.3 MB** (manifest 180 KB + 4096 centroids × 3.8 KB) |
| Insert throughput | 7,868 vec/s |
| Flush writes | **29.3 MB** (2866 delta segments, avg 10.5 KB each) |
| Old data reads during flush | **0** |
| Files deleted during flush | **0** |
| Delta/Base ratio | 3.4% |

### Multi-Round Delta Accumulation (128 clusters, 100K base)

```
Round 0: 128 base segments      → 100,000 vectors
Round 1: +121 delta segments    → 110,000 vectors (search ✅)
Round 2: +121 delta segments    → 120,000 vectors (search ✅)
```

All segments immutable, no data loss, recall consistent across rounds.

---

## Key Source Files

| File | Purpose |
|------|---------|
| [`src/quantizer.rs`](../src/quantizer.rs) | RaBitQ quantization and reconstruction |
| [`src/ivf.rs`](../src/ivf.rs) | IVF index, ClusterData, builder, search, flush |
| [`src/manifest.rs`](../src/manifest.rs) | V4 manifest + segment read/write (object_store) |
| [`src/kmeans.rs`](../src/kmeans.rs) | K-Means with reservoir sampling, faer GEMM |
| [`src/rotation.rs`](../src/rotation.rs) | FHT random rotation |
| [`src/simd.rs`](../src/simd.rs) | AVX2 FastScan kernels, ex-code inner product |
| [`src/math.rs`](../src/math.rs) | L2 distance, dot product |
| [`src/bin/ivf_rabitq.rs`](../src/bin/ivf_rabitq.rs) | CLI: streaming build / lazy load / insert / delta flush |
| [`src/bin/ivf_bench.rs`](../src/bin/ivf_bench.rs) | CLI: recall@k + QPS benchmark (FS / S3) |
