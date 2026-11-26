#pragma once

// Hand-offs between host and device through page-locked host memory the
// kernels reach directly, which skip the copy engines.

// out[i] = x[i] for the first n values; `out` is host memory.
extern "C" __global__ void to_host(float* out, const float* x, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i];
}
