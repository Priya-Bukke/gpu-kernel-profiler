# Results on Intel(R) Iris(R) Xe Graphics

Intel(R) Iris(R) Xe Graphics (ALL | GPU, 80 compute units at 1200 MHz, 64 KB local memory, 1536 GFLOP/s FP32 peak).

Measured memory bandwidth (best STREAM kernel): **32.8 GB/s**.
Ridge point of the roofline: 46.9 FLOP/byte. Kernels below it are limited by memory bandwidth, kernels above it by compute.

Speedup is against the first (v1) GPU version of the same suite. The NumPy rows run on the laptop's CPU.

| Suite | Variant | Size | Time | GB/s | GFLOP/s | % of roof | FLOP/byte | Speedup | Check |
|---|---|---|---|---|---|---|---|---|---|
| stream | copy | 16,777,216 floats | 4.352 ms | 30.8 | 0.0 | 94% of best | 0.00 |  | ok |
| stream | triad | 16,777,216 floats | 6.145 ms | 32.8 | 5.5 | 100% | 0.17 |  | ok |
| stream | triad float4 | 16,777,216 floats | 6.455 ms | 31.2 | 5.2 | 95% | 0.17 |  | ok |
| reduce | v1 interleaved | 16,777,216 floats | 6.193 ms | 10.8 | 2.7 | 33% | 0.25 | 1.0x | ok |
| reduce | v2 sequential | 16,777,216 floats | 4.204 ms | 16.0 | 4.0 | 49% | 0.25 | 1.5x | ok |
| reduce | v3 grid stride | 16,777,216 floats | 1.793 ms | 37.4 | 9.4 | 114% | 0.25 | 3.5x | ok |
| reduce | v4 subgroups | 16,777,216 floats | 1.789 ms | 37.5 | 9.4 | 114% | 0.25 | 3.5x | ok |
| reduce | NumPy on CPU (reference) | 16,777,216 floats | 8.233 ms | 8.2 | 2.0 | CPU | 0.25 |  | ok |
| matmul | v1 naive | 1024x1024 | 29.391 ms | 0.4 | 73.1 | 5% | 170.67 | 1.0x | ok |
| matmul | v2 local memory tiles | 1024x1024 | 15.099 ms | 0.8 | 142.2 | 9% | 170.67 | 1.9x | ok |
| matmul | v3 register blocking | 1024x1024 | 10.317 ms | 1.2 | 208.1 | 14% | 170.67 | 2.8x | ok |
| matmul | NumPy on CPU (reference) | 1024x1024 | 8.314 ms | 1.5 | 258.3 | CPU | 170.67 |  | ok |
| fir | v1 naive (64 taps) | 4,194,304 samples | 3.284 ms | 10.2 | 163.5 | 31% | 16.00 | 1.0x | ok |
| fir | v2 local window (64 taps) | 4,194,304 samples | 1.933 ms | 17.4 | 277.8 | 53% | 16.00 | 1.7x | ok |
| fir | NumPy on CPU (reference) | 4,194,304 samples | 60.281 ms | 0.6 | 8.9 | CPU | 16.00 |  | ok |

![roofline](roofline.png)

