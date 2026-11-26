#pragma once

#include "common.cuh"

// Attention over depth (Inkling 2): a mix of the outputs of earlier blocks
// in place of a residual stream, and the gate that takes a looped layer's
// message in.

// Must match depth.rs.
constexpr unsigned DEPTH_MAX_STATES = 8;
constexpr unsigned DEPTH_MAX_OUTS = 4;
constexpr float DEPTH_EPS = 1e-6f;

struct DepthMixArgs {
    // n rows of d values each.
    const float* states[DEPTH_MAX_STATES];
    float* outs[DEPTH_MAX_OUTS];
    // d values per query; output o takes query min(o, n_queries - 1).
    const float* queries;
    unsigned n_states;
    unsigned n_queries;
    unsigned n_outs;
    unsigned d;
};

// Per token and output: the states normalized by their root mean square,
// weighted by the softmax of their dot products with the query over the
// square root of d. grid n, block 256. Every thread runs through its values
// once for all the squares and dot products, and they are added up across
// the block together, each in the order `block_sum` adds one.
extern "C" __global__ void depth_mix(DepthMixArgs a) {
    constexpr unsigned SUMS = DEPTH_MAX_STATES * (1 + DEPTH_MAX_OUTS);
    __shared__ float part[WARP][SUMS];
    const size_t row = static_cast<size_t>(blockIdx.x) * a.d;
    // the squares of state s, then its dot products with each output's query
    float sums[SUMS] = {};
    for (unsigned k = threadIdx.x; k < a.d; k += blockDim.x) {
#pragma unroll
        for (unsigned s = 0; s < DEPTH_MAX_STATES; ++s) {
            if (s >= a.n_states) break;
            const float x = a.states[s][row + k];
            sums[s] += x * x;
#pragma unroll
            for (unsigned o = 0; o < DEPTH_MAX_OUTS; ++o) {
                if (o >= a.n_outs) break;
                const float* q = a.queries + static_cast<size_t>(min(o, a.n_queries - 1)) * a.d;
                sums[DEPTH_MAX_STATES * (1 + o) + s] += a.states[s][row + k] * q[k];
            }
        }
    }
    const unsigned warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
#pragma unroll
    for (unsigned i = 0; i < SUMS; ++i) {
        const float v = warp_sum(sums[i]);
        if (lane == 0) part[warp][i] = v;
    }
    __syncthreads();
#pragma unroll
    for (unsigned i = 0; i < SUMS; ++i) sums[i] = warp_sum(lane < blockDim.x / WARP ? part[lane][i] : 0.0f);

    float inv[DEPTH_MAX_STATES];
    for (unsigned s = 0; s < DEPTH_MAX_STATES; ++s) {
        if (s >= a.n_states) break;
        inv[s] = 1.0f / sqrtf(sums[s] / static_cast<float>(a.d) + DEPTH_EPS);
    }
    const float scale = 1.0f / sqrtf(static_cast<float>(a.d));
    for (unsigned o = 0; o < a.n_outs; ++o) {
        float p[DEPTH_MAX_STATES];
        float top = -CUDART_INF_F;
        for (unsigned s = 0; s < DEPTH_MAX_STATES; ++s) {
            if (s >= a.n_states) break;
            p[s] = sums[DEPTH_MAX_STATES * (1 + o) + s] * inv[s] * scale;
            top = fmaxf(top, p[s]);
        }
        float sum = 0.0f;
        for (unsigned s = 0; s < DEPTH_MAX_STATES; ++s) {
            if (s >= a.n_states) break;
            p[s] = expf(p[s] - top);
            sum += p[s];
        }
        for (unsigned s = 0; s < DEPTH_MAX_STATES; ++s) {
            if (s >= a.n_states) break;
            p[s] = p[s] / sum * inv[s];
        }
        for (unsigned k = threadIdx.x; k < a.d; k += blockDim.x) {
            float v = 0.0f;
            for (unsigned s = 0; s < DEPTH_MAX_STATES; ++s) {
                if (s >= a.n_states) break;
                v += p[s] * a.states[s][row + k];
            }
            a.outs[o][row + k] = v;
        }
    }
}

// x = x - tanh(forget) x + tanh(gain) message, elementwise over n values.
extern "C" __global__ void loop_control(float* x, const float* message, const float* gain,
                                        const float* forget, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = (v - tanhf(forget[i]) * v) + tanhf(gain[i]) * message[i];
}
