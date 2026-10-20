// Sum reduction, four versions. Each work group writes one partial sum; the host adds the partials.
//
// The work group size is read once into a variable before every barrier loop. Written directly in the loop
// condition (s < get_local_size(0)), PoCL 3.0 on the CPU produced wrong sums for v1 while the Intel GPU
// driver was fine; a plainly uniform loop bound is correct on both. CI runs these kernels on PoCL.

// v1: tree reduction in local memory with interleaved addressing. The modulo test leaves most work items
// idle inside each subgroup (divergence), and the stride pattern causes local memory bank conflicts.
__kernel void reduce_interleaved(__global const float *in, __global float *partial, __local float *scratch)
{
    const uint lid = get_local_id(0);
    const uint wg = get_local_size(0);
    scratch[lid] = in[get_global_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = 1; s < wg; s *= 2) {
        if (lid % (2 * s) == 0)
            scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0)
        partial[get_group_id(0)] = scratch[0];
}

// v2: sequential addressing. Active work items stay contiguous, so whole subgroups retire together.
__kernel void reduce_sequential(__global const float *in, __global float *partial, __local float *scratch)
{
    const uint lid = get_local_id(0);
    const uint wg = get_local_size(0);
    scratch[lid] = in[get_global_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = wg / 2; s > 0; s >>= 1) {
        if (lid < s)
            scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0)
        partial[get_group_id(0)] = scratch[0];
}

// v3: every work item first sums many elements with a grid stride loop (coalesced reads), so the
// launch needs far fewer work groups and the local memory tree runs far less often.
__kernel void reduce_grid_stride(__global const float *in, __global float *partial, __local float *scratch,
                                 const uint n)
{
    const uint lid = get_local_id(0);
    const uint wg = get_local_size(0);
    float acc = 0.0f;
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0))
        acc += in[i];
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = wg / 2; s > 0; s >>= 1) {
        if (lid < s)
            scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0)
        partial[get_group_id(0)] = scratch[0];
}

#ifdef HAS_SUBGROUPS
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
// v4: grid stride accumulation, then a subgroup reduction in registers (no barriers inside a subgroup),
// and local memory only to combine one value per subgroup.
__kernel void reduce_subgroup(__global const float *in, __global float *partial, __local float *scratch,
                              const uint n)
{
    float acc = 0.0f;
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0))
        acc += in[i];
    acc = sub_group_reduce_add(acc);
    if (get_sub_group_local_id() == 0)
        scratch[get_sub_group_id()] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0) {
        float total = 0.0f;
        for (uint s = 0; s < get_num_sub_groups(); s++)
            total += scratch[s];
        partial[get_group_id(0)] = total;
    }
}
#endif
