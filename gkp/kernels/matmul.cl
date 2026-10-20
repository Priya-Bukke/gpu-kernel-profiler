// C = A * B for square row major matrices of size N (N a multiple of the tile sizes).

// v1: one work item per output element, every operand read straight from global memory.
__kernel void matmul_naive(__global const float *A, __global const float *B, __global float *C, const int N)
{
    int row = get_global_id(1);
    int col = get_global_id(0);
    float acc = 0.0f;
    for (int k = 0; k < N; k++)
        acc += A[row * N + k] * B[k * N + col];
    C[row * N + col] = acc;
}

// v2: TILE x TILE work groups stage tiles of A and B in shared local memory, so each global value is read
// once per tile instead of once per multiply.
__kernel void matmul_tiled(__global const float *A, __global const float *B, __global float *C, const int N)
{
    __local float As[TILE][TILE];
    __local float Bs[TILE][TILE];
    int lr = get_local_id(1), lc = get_local_id(0);
    int row = get_global_id(1), col = get_global_id(0);
    float acc = 0.0f;
    for (int t = 0; t < N; t += TILE) {
        As[lr][lc] = A[row * N + t + lc];
        Bs[lr][lc] = B[(t + lr) * N + col];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE; k++)
            acc += As[lr][k] * Bs[k][lc];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[row * N + col] = acc;
}

// v3: register blocking. Each work item computes a WPT x 1 strip of outputs, reusing every value of B it
// loads from local memory WPT times from registers. Work group is (TILE, TILE / WPT).
__kernel void matmul_regblock(__global const float *A, __global const float *B, __global float *C, const int N)
{
    __local float As[TILE][TILE];
    __local float Bs[TILE][TILE];
    int lc = get_local_id(0);
    int lr = get_local_id(1);                 // 0 .. TILE/WPT - 1
    int col = get_group_id(0) * TILE + lc;
    int row0 = get_group_id(1) * TILE + lr;
    const int RTS = TILE / WPT;               // rows handled per pass of the work group
    float acc[WPT];
    for (int w = 0; w < WPT; w++)
        acc[w] = 0.0f;
    for (int t = 0; t < N; t += TILE) {
        for (int w = 0; w < WPT; w++) {
            As[lr + w * RTS][lc] = A[(row0 + w * RTS) * N + t + lc];
            Bs[lr + w * RTS][lc] = B[(t + lr + w * RTS) * N + col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE; k++) {
            float b = Bs[k][lc];
            for (int w = 0; w < WPT; w++)
                acc[w] += As[lr + w * RTS][k] * b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int w = 0; w < WPT; w++)
        C[(row0 + w * RTS) * N + col] = acc[w];
}
