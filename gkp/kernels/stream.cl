// STREAM style kernels: they do almost no arithmetic, so their speed is the memory bandwidth.

__kernel void copy(__global const float *a, __global float *c)
{
    size_t i = get_global_id(0);
    c[i] = a[i];
}

__kernel void triad(__global const float *a, __global const float *b, __global float *c, const float s)
{
    size_t i = get_global_id(0);
    c[i] = a[i] + s * b[i];
}

// Same triad with float4: fewer, wider memory transactions per work item.
__kernel void triad_vec4(__global const float4 *a, __global const float4 *b, __global float4 *c, const float s)
{
    size_t i = get_global_id(0);
    c[i] = a[i] + s * b[i];
}
