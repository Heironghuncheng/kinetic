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
// square root of d. grid n, block 256.
extern "C" __global__ void depth_mix(DepthMixArgs a) {
    const size_t row = static_cast<size_t>(blockIdx.x) * a.d;
    float inv[DEPTH_MAX_STATES];
    for (unsigned s = 0; s < DEPTH_MAX_STATES; ++s) {
        if (s >= a.n_states) break;
        float ss = 0.0f;
        for (unsigned k = threadIdx.x; k < a.d; k += blockDim.x) {
            const float x = a.states[s][row + k];
            ss += x * x;
        }
        inv[s] = 1.0f / sqrtf(block_sum(ss) / static_cast<float>(a.d) + DEPTH_EPS);
    }
    const float scale = 1.0f / sqrtf(static_cast<float>(a.d));
    for (unsigned o = 0; o < a.n_outs; ++o) {
        const float* q = a.queries + static_cast<size_t>(min(o, a.n_queries - 1)) * a.d;
        float p[DEPTH_MAX_STATES];
        float top = -CUDART_INF_F;
        for (unsigned s = 0; s < DEPTH_MAX_STATES; ++s) {
            if (s >= a.n_states) break;
            float dot = 0.0f;
            for (unsigned k = threadIdx.x; k < a.d; k += blockDim.x) dot += a.states[s][row + k] * q[k];
            p[s] = block_sum(dot) * inv[s] * scale;
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
