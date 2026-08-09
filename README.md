# GEMM from Scratch

Implementation and performance analysis of **General Matrix-Matrix Multiplication (GEMM)** developed from scratch using C and CUDA.

The project progressively optimizes matrix multiplication on an **NVIDIA Tesla T4**, starting from a sequential CPU implementation and moving through several CUDA optimization techniques:

* Sequential CPU matrix multiplication
* Naive CUDA kernel
* Shared-memory tiling
* Shared-memory padding
* Register tiling
* Vectorized loads with `float4`
* Read-only cache through `__ldg()`

The goal is to understand how memory hierarchy, data reuse, arithmetic intensity, and GPU execution affect GEMM performance.

---

## Overview

For two square matrices

$$
C = A \times B
$$

where (A, B, C \in \mathbb{R}^{N \times N}), the computation requires:

$$
2N^3
$$

floating-point operations.


This project studies how different implementations affect:

* Execution time
* TFLOP/s
* Arithmetic intensity
* Hardware efficiency
* Speedup
* Scaling with matrix size
* Memory traffic
* Roofline behavior

The experiments use matrix sizes:

```text
N = 256, 512, 1024, 2048, 4096
```

---

## Hardware

The experiments were performed on an:

**NVIDIA Tesla T4**

with the following reference specifications used by the project:

| Specification         |        Value |
| --------------------- | -----------: |
| FP32 peak performance | ~8.1 TFLOP/s |
| Memory bandwidth      |     320 GB/s |
| CUDA architecture     |      `sm_75` |

The notebook compiles CUDA kernels using:

```bash
-arch=sm_75
```

The project uses the T4 FP32 peak of **8.1 TFLOP/s** when calculating hardware efficiency.

---

## Optimization Pipeline

### 1. Sequential CPU

The first implementation is a sequential C matrix multiplication.

It serves as the correctness baseline and establishes the computational cost of GEMM without GPU parallelism.

The implementation performs the multiplication using CPU loops and transposes `B` to improve the sequential access pattern.

For large matrices, the arithmetic intensity remains approximately:

```text
0.25 FLOP/byte
```

---

### 2. Naive CUDA

The first GPU implementation assigns one CUDA thread to each output element.

Conceptually:

```text
thread → C[row][col]
```

Each thread computes the complete dot product:

```cpp
for (int k = 0; k < N; k++)
    sum += A[row * N + k] * B[k * N + col];
```

This introduces massive parallelism compared with the CPU implementation, but does not significantly improve data reuse.

Therefore, its arithmetic intensity remains approximately:

```text
0.25 FLOP/byte
```

The notebook shows a substantial improvement over the CPU baseline. At `N=1024`, for example, the naive CUDA kernel runs in approximately `5.00 ms` versus `1500.87 ms` for the optimized sequential CPU benchmark.

---

### 3. Shared-Memory Tiling

The next optimization divides the matrices into tiles and loads them into CUDA shared memory.

Two tile sizes are evaluated:

```text
TILE = 16
TILE = 32
```

The main idea is to reuse values loaded from global memory:

```text
Global Memory
      ↓
Shared Memory
      ↓
Multiple computations
```

This reduces global-memory traffic and increases arithmetic intensity.

For large matrices, the theoretical arithmetic intensity approaches:

```text
TILE = 16 → ~4 FLOP/byte
TILE = 32 → ~8 FLOP/byte
```

The measured values approach these theoretical limits.

The `TILE=32` implementation performs better than `TILE=16` in the experiments.

---

### 4. Shared-Memory Padding

The project then experiments with padding shared-memory arrays:

```cpp
__shared__ float As[TILE][TILE + 1];
__shared__ float Bs[TILE][TILE + 1];
```

The purpose is to eliminate potential shared-memory bank conflicts.

However, the benchmark demonstrates that padding **does not improve this particular kernel**.

At `N=1024`:

```text
Tiled TILE=32:   3.984 ms
Padded TILE=32:  6.834 ms
```

The original access pattern was already conflict-free, so padding adds overhead without removing an actual bottleneck.

This is an important result of the experiment: an optimization that is theoretically useful is not necessarily beneficial for every access pattern.

---

### 5. Register Tiling

Register tiling extends shared-memory tiling by giving each thread a small output micro-tile.

The configuration used is:

```text
TILE = 64
TW   = 4
```

Therefore, each thread computes:

```text
4 × 4 = 16 output values
```

The partial results are accumulated in registers:

```cpp
float regC[TW][TW];
```

This increases data reuse and reduces the relative cost of shared/global memory accesses.

The theoretical arithmetic intensity approaches:

```text
16 FLOP/byte
```

and the notebook measurements approach this value as `N` increases.

---

### 6. Vectorized Loads + Read-Only Cache

The final optimization combines:

* `float4` vectorized loads
* CUDA `__ldg()`
* Register tiling
* Shared memory

Instead of loading one `float` at a time, the kernel loads four values:

```cpp
float4 a4 = __ldg((const float4 *)&A[...]);
```

This allows each load instruction to transfer four FP32 values.

The `__ldg()` intrinsic is used to access data through the read-only cache.

Importantly, this optimization does **not** increase arithmetic intensity. Instead, it improves the efficiency of the memory-access path and reduces instruction overhead.

---

## Results

The benchmark results stored in the notebook show the following performance at `N=4096`:

| Implementation      | Time (ms) |    TFLOP/s | Arithmetic Intensity |
| ------------------- | --------: | ---------: | -------------------: |
| Sequential CPU      | 99,496.17 |     0.0015 |                0.250 |
| Naive CUDA          |    229.66 |     0.5984 |                0.250 |
| Tiled `T=16`        |    201.84 |     0.6809 |                3.992 |
| Tiled `T=32`        |    152.94 |     0.8987 |                7.969 |
| Padded `T=16`       |    252.26 |     0.5448 |                3.992 |
| Padded `T=32`       |    192.90 |     0.7125 |                7.969 |
| Register tiled      |     65.13 |     2.1102 |               15.876 |
| **Vec + `__ldg()`** | **58.35** | **2.3552** |           **15.876** |

All implementations passed the correctness checks included in the benchmark pipeline.

The progression clearly shows the relationship between data reuse and performance: increasing arithmetic intensity through tiling and register blocking produces increasingly higher throughput.

### Performance at N = 1024

For the final notebook benchmark:

```text
Register tiled:
0.8700 TFLOP/s

Vec + __ldg():
1.0320 TFLOP/s
```

The vectorized version therefore provides an additional improvement over register tiling at this matrix size.

---

## Arithmetic Intensity

For the naive implementation, the project models the memory traffic as approximately:

[
(2N^3 + N^2)\times4
]

bytes for FP32 values.

Therefore, for large `N`:

[
AI \approx 0.25 \text{ FLOP/byte}
]

This low arithmetic intensity explains why the naive implementation is strongly affected by memory traffic.

With shared-memory tiling:

[
AI_{tiled} \approx \frac{TILE}{4}
]

giving approximately:

```text
TILE=16 → 4 FLOP/byte
TILE=32 → 8 FLOP/byte
```

Register tiling with `TILE=64` increases the theoretical value to approximately:

```text
16 FLOP/byte
```

The notebook measurements confirm this progression.

---

## Project Structure

The notebook generates the C/CUDA source files used by the experiments.

The main components are:

```text
.
├── gemm-from-scratch.ipynb
├── helpers.h
├── helpers.c
├── matmul_seq.c
├── matmul_naive.cu
├── matmul_tiled.cu
├── matmul_padded.cu
├── matmul_regtile.cu
├── matmul_vec_ldg.cu
├── refs/
├── figures/
└── results.csv
```

The generated source files contain the individual GEMM implementations, while the notebook provides the benchmark framework, compilation commands, validation, data collection, and visualization.

---

## Requirements

### Hardware

A CUDA-capable NVIDIA GPU is required.

The experiments in this repository target:

```text
NVIDIA Tesla T4
Compute Capability 7.5
```

### Software

The notebook expects:

* GCC
* NVIDIA CUDA Toolkit / `nvcc`
* Python 3
* NumPy
* pandas
* Matplotlib
* Jupyter Notebook or JupyterLab

The notebook checks the available toolchain using:

```bash
gcc --version
nvcc --version
nvidia-smi
```

---

## Running the Notebook

Start Jupyter:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

Then open:

```text
gemm-from-scratch.ipynb
```

The notebook automatically generates the C and CUDA source files using `%%writefile`.

---

## Compiling the Implementations

### Sequential CPU

The notebook compiles the CPU implementation with:

```bash
gcc -O2 -o matmul_seq matmul_seq.c helpers.c -lm
```

### Naive CUDA

```bash
nvcc -O2 -arch=sm_75 -o matmul_naive matmul_naive.cu helpers.c
```

### Tiled GEMM — TILE 16

```bash
nvcc -O2 -arch=sm_75 -DTILE=16 \
    -o matmul_tiled16 matmul_tiled.cu helpers.c
```

### Tiled GEMM — TILE 32

```bash
nvcc -O2 -arch=sm_75 -DTILE=32 \
    -o matmul_tiled32 matmul_tiled.cu helpers.c
```

### Padded GEMM

```bash
nvcc -O2 -arch=sm_75 -DTILE=16 \
    -o matmul_padded16 matmul_padded.cu helpers.c

nvcc -O2 -arch=sm_75 -DTILE=32 \
    -o matmul_padded32 matmul_padded.cu helpers.c
```

### Register Tiling

```bash
nvcc -O2 -arch=sm_75 -DTILE=64 -DTW=4 \
    -o matmul_regtile matmul_regtile.cu helpers.c
```

### Vectorized Loads + `__ldg()`

```bash
nvcc -O2 -arch=sm_75 -DTILE=64 -DTW=4 \
    -o matmul_vec_ldg matmul_vec_ldg.cu helpers.c
```

---

## Running Benchmarks

The executables accept a reference directory followed by matrix sizes.

For example:

```bash
./matmul_seq ./refs/ 256 512 1024 2048 4096
```

For the naive CUDA kernel:

```bash
./matmul_naive ./refs/ 256 512 1024 2048 4096
```

The same pattern applies to the other implementations.

The benchmark runner:

1. Executes each matrix size.
2. Measures kernel execution time.
3. Computes TFLOP/s.
4. Computes arithmetic intensity.
5. Calculates an output checksum.
6. Compares the result against the CPU reference.
7. Stores the results in `results.csv`.

Each benchmark is executed multiple times and uses the median execution time.

---

## Correctness Verification

The sequential CPU implementation generates reference outputs:

```text
refs/ref_output_<N>.bin
```

GPU implementations compare their output checksum against the corresponding CPU reference.

The notebook uses a relative tolerance when comparing checksums.

This allows every CUDA optimization to be evaluated not only by performance but also by correctness.

---

## Performance Analysis

The notebook generates plots for:

* TFLOP/s at a fixed matrix size
* TFLOP/s versus matrix size
* Execution time versus matrix size
* Speedup relative to the naive CUDA implementation
* Arithmetic intensity
* Hardware efficiency
* Roofline analysis

The generated figures are stored in:

```text
figures/
```

The analysis shows that GPU performance improves as matrix size increases because fixed costs such as kernel launches, synchronization, and memory latency become easier to amortize over a larger computational workload.

---

## Key Findings

### GPU parallelism provides the first major improvement

Moving from sequential CPU execution to the naive CUDA implementation produces a dramatic reduction in execution time.

At `N=1024`, the notebook reports approximately:

```text
CPU:         1500.87 ms
Naive CUDA:     5.00 ms
```

This demonstrates the impact of GPU parallelism even before applying memory optimizations.

### Tiling increases arithmetic intensity

Shared-memory tiling increases data reuse and reduces the amount of global-memory traffic required per FLOP.

```text
Naive:       ~0.25 FLOP/byte
TILE=16:     ~4.00 FLOP/byte
TILE=32:     ~8.00 FLOP/byte
```

### Padding is not always beneficial

The padded implementations were slower than their non-padded counterparts.

This happens because the original access pattern in this GEMM kernel already avoids the relevant bank conflicts, so padding increases shared-memory usage without solving an actual bottleneck.

### Register tiling provides a substantial improvement

Register tiling increases reuse inside each thread by computing a `4×4` micro-tile.

The notebook reports:

```text
N=4096
Register tiling → 2.1102 TFLOP/s
```

### Vectorized loads improve the final implementation

Using `float4` loads together with `__ldg()` further improves the register-tiled implementation.

The notebook reports:

```text
N=4096
Vec + __ldg() → 2.3552 TFLOP/s
```

This corresponds to approximately:

```text
29.1% of the 8.1 TFLOP/s T4 FP32 peak
```

---

## Roofline Perspective

The roofline analysis summarizes the optimization process:

```text
Low AI
  │
  ├── Sequential CPU
  ├── Naive CUDA
  │
  ├── Shared-memory tiling
  │
  ├── Register tiling
  │
  └── Vectorized loads + __ldg()
          │
          ▼
      Higher AI
```

The initial implementations are dominated by memory traffic and limited data reuse.

Tiling increases reuse through shared memory, while register tiling further increases reuse by keeping intermediate values in registers.

The final vectorized implementation does not substantially change arithmetic intensity; instead, it improves the efficiency of the memory-access path.

---

## Results vs. Theoretical Peak

Even the best implementation remains below the theoretical FP32 peak of the Tesla T4.

This indicates that there is still room for architecture-specific optimization involving:

* More aggressive instruction-level optimization
* Improved memory access patterns
* Further register/shared-memory tuning
* Tensor Core based approaches
* Mixed-precision WMMA kernels

The original report identifies Tensor Core WMMA as a potential next step for increasing throughput on supported hardware.

---

## Benchmark Note

The repository contains both the notebook and a generated performance report.

There is a small difference between the benchmark snapshots:

* The **notebook execution** reports `2.3552 TFLOP/s` for `Vec + __ldg()` at `N=4096`.
* The **PDF report** reports `2.1424 TFLOP/s` for register tiling and `1.9988 TFLOP/s` for `Vec + __ldg()` at `N=4096`.

Therefore, the README uses the **notebook's recorded benchmark results** for the main results table rather than combining values from the two runs.

---

## What This Project Demonstrates

This project is primarily an educational exploration of GPU performance optimization.

It demonstrates how a mathematically identical GEMM operation can exhibit very different performance depending on:

* Parallel execution
* Global-memory traffic
* Shared-memory reuse
* Register reuse
* Memory access patterns
* Instruction overhead
* Arithmetic intensity
* GPU occupancy
* Hardware characteristics

The central lesson is that improving GPU performance is not simply about performing more operations in parallel. Efficient data movement and reuse are equally important.

---

## References

* NVIDIA Tesla T4 specifications: [TechPowerUp — NVIDIA Tesla T4](https://www.techpowerup.com/gpu-specs/tesla-t4.c3316)
* Project notebook: `gemm-from-scratch.ipynb`
* Performance report: `GEMM_from_Scratch(1).pdf`

---

## Author

**Juan Esteban Garcia Pulgarin**

April 2026
