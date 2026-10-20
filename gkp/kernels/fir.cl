// FIR filter y[n] = sum_k h[k] * x[n + k] (valid part only, len(y) = len(x) - TAPS + 1).

// v1: every work item reads its TAPS inputs and taps from global memory.
__kernel void fir_naive(__global const float *x, __constant float *h, __global float *y)
{
    size_t n = get_global_id(0);
    float acc = 0.0f;
    for (int k = 0; k < TAPS; k++)
        acc += h[k] * x[n + k];
    y[n] = acc;
}

// v2: the work group loads the input window it needs (WG + TAPS - 1 samples) into local memory once,
// then every work item filters from there. Neighbouring outputs share TAPS - 1 inputs.
__kernel void fir_local(__global const float *x, __constant float *h, __global float *y, __local float *win)
{
    size_t lid = get_local_id(0);
    size_t wg = get_local_size(0);
    size_t base = get_group_id(0) * wg;
    for (size_t i = lid; i < wg + TAPS - 1; i += wg)
        win[i] = x[base + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    float acc = 0.0f;
    for (int k = 0; k < TAPS; k++)
        acc += h[k] * win[lid + k];
    y[base + lid] = acc;
}
