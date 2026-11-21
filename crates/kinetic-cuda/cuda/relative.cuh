#pragma once

#include "quantize.cuh"

// Attention by relative distance (Inkling 2) and its residual short
// convolutions. Every token's values depend only on its own inputs, the
// cache and the convolution rings, never on the rest of its batch, and the
// sums run in an order fixed by positions alone.

constexpr unsigned REL_THREADS = 128;
constexpr unsigned REL_WARPS = REL_THREADS / WARP;
// Keys per chunk; part j of a head attends the chunks c with
// c % REL_PARTS == j. Must match relative.rs.
constexpr unsigned REL_CHUNK = 64;
constexpr unsigned REL_PARTS = 32;
// Largest head and most relative terms per head.
constexpr unsigned REL_LANE_VALUES = 8;
constexpr unsigned REL_MAX_REL = 32;

struct RelConvArgs {
    // [first position, token ids...]
    const unsigned* batch;
    // n rows of `channels` inputs.
    const float* x;
    // `taps` weights per channel, oldest input first.
    const float* w;
    // Per row, `taps` weights per run of `group` channels, newest input
    // first; null without.
    const float* dynamic;
    // The inputs, row `pos % ring_len` holding position `pos`.
    float* ring;
    float* y;
    unsigned n;
    unsigned channels;
    unsigned taps;
    unsigned ring_len;
    unsigned group;
    float scale;
};

__device__ __forceinline__ float rel_conv_input(const RelConvArgs& a, int p, int pos0, unsigned ch) {
    if (p >= pos0) return a.x[static_cast<size_t>(p - pos0) * a.channels + ch];
    if (p < 0) return 0.0f;
    return a.ring[static_cast<size_t>(p % static_cast<int>(a.ring_len)) * a.channels + ch];
}

// y = (x + conv(x)) * scale, inputs before the batch from the ring.
// grid (ceil(channels / 128), n), block 128.
extern "C" __global__ void rel_conv(RelConvArgs a) {
    const unsigned ch = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (ch >= a.channels) return;
    const int pos0 = static_cast<int>(a.batch[0]), pos = pos0 + static_cast<int>(t);
    const float* w = a.w + static_cast<size_t>(ch) * a.taps;
    float acc = 0.0f;
    for (unsigned i = 0; i < a.taps; ++i)
        acc = fmaf(w[i], rel_conv_input(a, pos - static_cast<int>(a.taps - 1 - i), pos0, ch), acc);
    if (a.dynamic) {
        const size_t heads = a.channels / a.group;
        const float* lags = a.dynamic + (t * heads + ch / a.group) * a.taps;
        for (unsigned lag = 0; lag < a.taps; ++lag)
            acc = fmaf(lags[lag], rel_conv_input(a, pos - static_cast<int>(lag), pos0, ch), acc);
    }
    const size_t at = static_cast<size_t>(t) * a.channels + ch;
    a.y[at] = (a.x[at] + acc) * a.scale;
}

// grid ceil(channels / 128), block 128: the batch's inputs into the ring,
// after `rel_conv` has read the older ones.
extern "C" __global__ void rel_conv_store(RelConvArgs a) {
    const unsigned ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= a.channels) return;
    const unsigned pos0 = a.batch[0];
    for (unsigned t = a.n > a.ring_len ? a.n - a.ring_len : 0; t < a.n; ++t)
        a.ring[static_cast<size_t>((pos0 + t) % a.ring_len) * a.channels + ch] =
            a.x[static_cast<size_t>(t) * a.channels + ch];
}

struct RelPrepareArgs {
    const unsigned* batch;
    // Per token: n_head query heads, n_head relative runs, n_head_kv key and
    // value heads.
    float* q;
    float* r;
    float* k;
    float* v;
    const float* q_scale;
    const float* rel_scale;
    __half* k_cache;
    __half* v_cache;
    unsigned n_head;
    unsigned n_head_kv;
    unsigned head_dim;
    unsigned n_rel;
    unsigned slots;
    float qk_clip;
    float rel_clip;
    // tau = 1 + alpha ln(max(1, (pos + 1) floor_inv)); alpha 0 leaves 1.
    float alpha;
    float floor_inv;
    float eps;
};

// Scales one warp's run of `len` values of x (a lane every 32) by `scale`,
// then down to a norm of at most `max`, then by `tau`.
__device__ __forceinline__ void rel_clip(float* x, const float* scale, unsigned len, float max,
                                         float tau, unsigned lane) {
    float ss = 0.0f;
    for (unsigned j = lane; j < len; j += WARP) {
        if (scale) x[j] *= scale[j];
        ss += x[j] * x[j];
    }
    const float norm = sqrtf(warp_sum(ss)) + 1e-6f;
    const float f = fminf(fmaxf(max / norm, 0.0f), 1.0f);
    for (unsigned j = lane; j < len; j += WARP) x[j] = x[j] * f * tau;
}

// One block per token: clips the query heads and relative terms, the key
// heads, normalizes the value heads and stores keys and values in the
// cache. grid n, block 128.
extern "C" __global__ void __launch_bounds__(REL_THREADS) rel_prepare(RelPrepareArgs a) {
    const unsigned t = blockIdx.x, warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const unsigned pos = a.batch[0] + t, hd = a.head_dim, kv_dim = a.n_head_kv * hd;
    const float tau = 1.0f + a.alpha * logf(fmaxf((static_cast<float>(pos) + 1.0f) * a.floor_inv, 1.0f));
    for (unsigned h = warp; h < a.n_head; h += REL_WARPS) {
        const size_t head = static_cast<size_t>(t) * a.n_head + h;
        rel_clip(a.q + head * hd, a.q_scale + h * hd, hd, a.qk_clip, tau, lane);
        rel_clip(a.r + head * a.n_rel, a.rel_scale + h * a.n_rel, a.n_rel, a.rel_clip, tau, lane);
    }
    const size_t slot = static_cast<size_t>(pos % a.slots) * kv_dim;
    for (unsigned h = warp; h < a.n_head_kv; h += REL_WARPS) {
        float* k = a.k + static_cast<size_t>(t) * kv_dim + h * hd;
        float* v = a.v + static_cast<size_t>(t) * kv_dim + h * hd;
        rel_clip(k, nullptr, hd, a.qk_clip, 1.0f, lane);
        float ss = 0.0f;
        for (unsigned j = lane; j < hd; j += WARP) ss += v[j] * v[j];
        const float inv = 1.0f / sqrtf(warp_sum(ss) / static_cast<float>(hd) + a.eps);
        for (unsigned j = lane; j < hd; j += WARP) {
            v[j] *= inv;
            a.k_cache[slot + h * hd + j] = __float2half(k[j]);
            a.v_cache[slot + h * hd + j] = __float2half(v[j]);
        }
    }
}

struct RelAttendArgs {
    // The position of the step's first token.
    const unsigned* pos;
    const float* q;
    const float* r;
    const __half* k_cache;
    const __half* v_cache;
    // `extent` rows of n_rel weights, by distance.
    const float* rel_bias;
    // Per token, head and part: the largest logit, the sum of exponentials
    // and head_dim weighted values.
    float* partial;
    unsigned n_head;
    unsigned n_head_kv;
    unsigned head_dim;
    unsigned n_rel;
    unsigned slots;
    // 0 for none.
    unsigned window;
    unsigned extent;
    float scale;
};

// Each warp keeps a running softmax over its keys of every chunk of the
// block's part, the warps then merge in order. grid (REL_PARTS, n_head, n),
// block 128.
extern "C" __global__ void __launch_bounds__(REL_THREADS) rel_attend(RelAttendArgs a) {
    __shared__ float rs[REL_MAX_REL];
    __shared__ float merged[REL_WARPS][2 + REL_LANE_VALUES * WARP];
    const unsigned part = blockIdx.x, h = blockIdx.y, t = blockIdx.z;
    const unsigned warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const unsigned hd = a.head_dim, per = hd / WARP;
    const unsigned pos = a.pos[0] + t;
    const unsigned lo = a.window && pos + 1 > a.window ? pos + 1 - a.window : 0;
    const size_t head = static_cast<size_t>(t) * a.n_head + h;
    float q[REL_LANE_VALUES];
#pragma unroll
    for (unsigned i = 0; i < REL_LANE_VALUES; ++i) q[i] = i < per ? a.q[head * hd + lane + WARP * i] : 0.0f;
    if (threadIdx.x < a.n_rel) rs[threadIdx.x] = a.r[head * a.n_rel + threadIdx.x];
    __syncthreads();

    const size_t kv_dim = static_cast<size_t>(a.n_head_kv) * hd;
    const size_t kv_off = static_cast<size_t>(h / (a.n_head / a.n_head_kv)) * hd;
    float m = -CUDART_INF_F, l = 0.0f, o[REL_LANE_VALUES] = {};
    const unsigned first = lo / REL_CHUNK, last = pos / REL_CHUNK;
    for (unsigned c = first + (part + REL_PARTS - first % REL_PARTS) % REL_PARTS; c <= last; c += REL_PARTS) {
        for (unsigned i = 0; i < REL_CHUNK / REL_WARPS; ++i) {
            const unsigned p = c * REL_CHUNK + warp * (REL_CHUNK / REL_WARPS) + i;
            if (p < lo || p > pos) continue;
            const size_t at = static_cast<size_t>(p % a.slots) * kv_dim + kv_off;
            float dot = 0.0f;
#pragma unroll
            for (unsigned j = 0; j < REL_LANE_VALUES; ++j)
                if (j < per) dot = fmaf(q[j], __half2float(a.k_cache[at + lane + WARP * j]), dot);
            dot = warp_sum(dot);
            const unsigned dist = pos - p;
            float bias = 0.0f;
            if (dist < a.extent) {
                const float* b = a.rel_bias + static_cast<size_t>(dist) * a.n_rel;
                for (unsigned j = 0; j < a.n_rel; ++j) bias = fmaf(b[j], rs[j], bias);
            }
            const float s = dot * a.scale + bias;
            const float top = fmaxf(m, s);
            const float keep = expf(m - top), e = expf(s - top);
            l = l * keep + e;
#pragma unroll
            for (unsigned j = 0; j < REL_LANE_VALUES; ++j)
                if (j < per) o[j] = o[j] * keep + e * __half2float(a.v_cache[at + lane + WARP * j]);
            m = top;
        }
    }
    merged[warp][0] = m;
    merged[warp][1] = l;
    for (unsigned j = 0; j < per; ++j) merged[warp][2 + lane + WARP * j] = o[j];
    __syncthreads();
    if (warp != 0) return;
    float top = -CUDART_INF_F;
    for (unsigned w = 0; w < REL_WARPS; ++w)
        if (merged[w][1] > 0.0f) top = fmaxf(top, merged[w][0]);
    float sum = 0.0f, out[REL_LANE_VALUES] = {};
    for (unsigned w = 0; w < REL_WARPS; ++w) {
        if (!(merged[w][1] > 0.0f)) continue;
        const float e = expf(merged[w][0] - top);
        sum += merged[w][1] * e;
        for (unsigned j = 0; j < per; ++j) out[j] += merged[w][2 + lane + WARP * j] * e;
    }
    float* dst = a.partial + (head * REL_PARTS + part) * (hd + 2);
    if (lane == 0) {
        dst[0] = top;
        dst[1] = sum;
    }
    for (unsigned j = 0; j < per; ++j) dst[2 + lane + WARP * j] = out[j];
}

struct RelCombineArgs {
    const float* partial;
    // Per token and head: the sink logit; per token and value: the gate.
    const float* sink;
    const float* gate;
    Act out;
    unsigned n_head;
};

// Merges a head's parts in order with its sink, a logit whose value is
// zero, and gates the result. grid (n_head, n), block head_dim.
extern "C" __global__ void rel_combine(RelCombineArgs a) {
    const unsigned h = blockIdx.x, t = blockIdx.y, j = threadIdx.x, hd = blockDim.x;
    const size_t head = static_cast<size_t>(t) * a.n_head + h;
    const float* p = a.partial + head * REL_PARTS * (hd + 2);
    const float sink = a.sink[head];
    float top = sink;
    for (unsigned i = 0; i < REL_PARTS; ++i)
        if (p[i * (hd + 2) + 1] > 0.0f) top = fmaxf(top, p[i * (hd + 2)]);
    float sum = expf(sink - top), out = 0.0f;
    for (unsigned i = 0; i < REL_PARTS; ++i) {
        const float* part = p + i * (hd + 2);
        if (!(part[1] > 0.0f)) continue;
        const float e = expf(part[0] - top);
        sum += part[1] * e;
        out += part[2 + j] * e;
    }
    const size_t at = head * hd + j;
    a.out.put(at, out / sum * sigmoid(a.gate[at]));
}
