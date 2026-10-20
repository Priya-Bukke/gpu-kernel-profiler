# Results on Intel(R) Iris(R) Xe Graphics

Intel(R) Iris(R) Xe Graphics (ALL | GPU, 80 compute units at 1200 MHz, 64 KB local memory, 1536 GFLOP/s FP32 peak).

Measured memory bandwidth (best STREAM kernel): **32.7 GB/s**.
Ridge point of the roofline: 47.0 FLOP/byte. Kernels below it are limited by memory bandwidth, kernels above it by compute.

Speedup is against the first (v1) GPU version of the same suite. The NumPy rows run on the laptop's CPU.

| Suite | Variant | Size | Time | GB/s | GFLOP/s | % of roof | FLOP/byte | Speedup | Check |
|---|---|---|---|---|---|---|---|---|---|
| stream | copy | 16,777,216 floats | 4.345 ms | 30.9 | 0.0 | 95% of best | 0.00 |  | ok |
| stream | triad | 16,777,216 floats | 6.166 ms | 32.7 | 5.4 | 100% | 0.17 |  | ok |
| stream | triad float4 | 16,777,216 floats | 6.631 ms | 30.4 | 5.1 | 93% | 0.17 |  | ok |
| reduce | v1 interleaved | 16,777,216 floats | 4.961 ms | 13.5 | 3.4 | 41% | 0.25 | 1.0x | ok |
| reduce | v2 sequential | 16,777,216 floats | 3.223 ms | 20.8 | 5.2 | 64% | 0.25 | 1.5x | ok |
| reduce | v3 grid stride | 16,777,216 floats | 1.783 ms | 37.6 | 9.4 | 115% | 0.25 | 2.8x | ok |
| reduce | v4 subgroups | 16,777,216 floats | 1.768 ms | 38.0 | 9.5 | 116% | 0.25 | 2.8x | ok |
| reduce | NumPy on CPU (reference) | 16,777,216 floats | 11.593 ms | 5.8 | 1.4 | CPU | 0.25 |  | ok |
| matmul | v1 naive | 1024x1024 | 29.855 ms | 0.4 | 71.9 | 5% | 170.67 | 1.0x | ok |
| matmul | v2 local memory tiles | 1024x1024 | 15.221 ms | 0.8 | 141.1 | 9% | 170.67 | 2.0x | ok |
| matmul | v3 register blocking | 1024x1024 | 10.358 ms | 1.2 | 207.3 | 13% | 170.67 | 2.9x | ok |
| matmul | NumPy on CPU (reference) | 1024x1024 | 9.157 ms | 1.4 | 234.5 | CPU | 170.67 |  | ok |
| fir | v1 naive (64 taps) | 4,194,304 samples | 3.353 ms | 10.0 | 160.1 | 31% | 16.00 | 1.0x | ok |
| fir | v2 local window (64 taps) | 4,194,304 samples | 1.981 ms | 16.9 | 270.9 | 52% | 16.00 | 1.7x | ok |
| fir | NumPy on CPU (reference) | 4,194,304 samples | 46.388 ms | 0.7 | 11.6 | CPU | 16.00 |  | ok |

![roofline](roofline.png)

