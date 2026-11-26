#pragma once

#include "common.cuh"

// Routed experts read straight from an arena of cache slots in their file
// layout; each slot holds one expert's gate, up and down matrices.
//
// Every value is computed exactly as the host computes it (see
// kinetic-models' blocks/moe/kernels.rs): inputs are quantized per 256
// values, each block of a row gives an exact integer sum, and the blocks are
// added in order, each times the product of the two scales, less, in the
// K-quants with offsets, the exact integer sum of the offsets times the
// inputs, times the offset scale and the input scale, with every rounding
// spelled out. A row's value is then the same bits on the host and
// on the device, so where an expert runs never changes the output.

constexpr unsigned QK_IQ = 256;
// Most assignments of one expert a block multiplies at once.
constexpr unsigned MOE_GROUP = 8;
constexpr unsigned MOE_THREADS = 256;

// Byte offsets of the codebooks in the table buffer, read through the
// cache: a few KiB that every warp looks up at random.
constexpr unsigned IQ2XXS_OFFSET = 0;
constexpr unsigned IQ2XS_OFFSET = 2048;
constexpr unsigned IQ3XXS_OFFSET = 6144;
constexpr unsigned IQ4NL_OFFSET = 7168;
constexpr unsigned MXFP4_OFFSET = 7184;
constexpr unsigned IQ2S_OFFSET = 7200;
constexpr unsigned IQ1S_OFFSET = IQ2S_OFFSET + 1024 * 8;

// x quantized per 256 values: d = max|x| / 127, q = round(x / d) half away
// from zero. One warp per block, eight values per lane.
extern "C" __global__ void quantize_q8k(const float* x, int8_t* qs, float* ds, unsigned blocks) {
    const unsigned b = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (b >= blocks) return;
    const unsigned lane = threadIdx.x % WARP;
    const float* v = x + static_cast<size_t>(b) * QK_IQ + 8 * lane;
    float amax = 0.0f;
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(v[j]));
#pragma unroll
    for (unsigned o = WARP / 2; o > 0; o /= 2)
        amax = fmaxf(amax, __shfl_xor_sync(FULL_MASK, amax, o));
    const float d = __fdiv_rn(amax, 127.0f);
    const float inv = d == 0.0f ? 0.0f : __fdiv_rn(1.0f, d);
    int8_t* q = qs + static_cast<size_t>(b) * QK_IQ + 8 * lane;
#pragma unroll
    for (unsigned j = 0; j < 8; ++j)
        q[j] = static_cast<int8_t>(fminf(fmaxf(roundf(__fmul_rn(v[j], inv)), -127.0f), 127.0f));
    if (lane == 0) ds[b] = d;
}

// Four magnitudes in `v` negated where the low four bits of `signs` say.
__device__ __forceinline__ int signed4(unsigned v, unsigned signs) {
    const unsigned m = ((signs & 1u) ? 0xffu : 0u) | ((signs & 2u) ? 0xff00u : 0u) |
                       ((signs & 4u) ? 0xff0000u : 0u) | ((signs & 8u) ? 0xff000000u : 0u);
    return static_cast<int>(__vsub4(v ^ m, m));
}

__device__ __forceinline__ float half_at(const uint8_t* p) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(p[0] | p[1] << 8)));
}

__device__ __forceinline__ unsigned u32_at(const uint8_t* p) {
    return p[0] | p[1] << 8 | p[2] << 16 | static_cast<unsigned>(p[3]) << 24;
}

// A lane's eight weights of a block as signed bytes, with the integer
// multiplier and offset of the 16 or 32 values they belong to.
struct Lane8 {
    int w[2];
    int mul;
    int min;
};

__device__ __forceinline__ unsigned nibbles(const uint8_t* p, unsigned shift) {
    return u32_at(p) >> shift & 0x0f0f0f0fu;
}

// Each lane takes values 8 * lane .. 8 * lane + 8 of a 256-value block; `D`
// is the block scale the host folds its power of two into.
struct Iq2Xxs {
    static constexpr unsigned BYTES = 66;
    static constexpr bool MINS = false;
    static __device__ __forceinline__ float min_scale(const uint8_t*) { return 0.0f; }
    static constexpr unsigned TABLE = IQ2XXS_OFFSET;
    static __device__ __forceinline__ float scale(const uint8_t* b) {
        return __fmul_rn(half_at(b), 0.125f);
    }
    static __device__ __forceinline__ Lane8 lane(const uint8_t* b, unsigned lane,
                                                 const uint8_t* table) {
        const unsigned sub = lane / 4, l = lane % 4;
        const unsigned aux = u32_at(b + 6 + 8 * sub);
        const unsigned signs = iq_signs(aux >> (7 * l) & 127);
        const uint2 g = reinterpret_cast<const uint2*>(table)[b[2 + 8 * sub + l]];
        return {{signed4(g.x, signs), signed4(g.y, signs >> 4)}, static_cast<int>(2 * (aux >> 28) + 1), 0};
    }
};

struct Iq2Xs {
    static constexpr unsigned BYTES = 74;
    static constexpr bool MINS = false;
    static __device__ __forceinline__ float min_scale(const uint8_t*) { return 0.0f; }
    static constexpr unsigned TABLE = IQ2XS_OFFSET;
    static __device__ __forceinline__ float scale(const uint8_t* b) {
        return __fmul_rn(half_at(b), 0.125f);
    }
    static __device__ __forceinline__ Lane8 lane(const uint8_t* b, unsigned lane,
                                                 const uint8_t* table) {
        const unsigned q = b[2 + 2 * lane] | b[3 + 2 * lane] << 8;
        const unsigned s = b[66 + lane / 4] >> (4 * (lane / 2 % 2)) & 0xf;
        const unsigned signs = iq_signs(q >> 9);
        const uint2 g = reinterpret_cast<const uint2*>(table)[q & 511];
        return {{signed4(g.x, signs), signed4(g.y, signs >> 4)}, static_cast<int>(2 * s + 1), 0};
    }
};

struct Iq3Xxs {
    static constexpr unsigned BYTES = 98;
    static constexpr bool MINS = false;
    static __device__ __forceinline__ float min_scale(const uint8_t*) { return 0.0f; }
    static constexpr unsigned TABLE = IQ3XXS_OFFSET;
    static __device__ __forceinline__ float scale(const uint8_t* b) {
        return __fmul_rn(half_at(b), 0.25f);
    }
    static __device__ __forceinline__ Lane8 lane(const uint8_t* b, unsigned lane,
                                                 const uint8_t* table) {
        const unsigned sub = lane / 4, l = lane % 4;
        const unsigned aux = u32_at(b + 66 + 4 * sub);
        const unsigned signs = iq_signs(aux >> (7 * l) & 127);
        const unsigned* grid = reinterpret_cast<const unsigned*>(table);
        const unsigned lo = grid[b[2 + 8 * sub + 2 * l]], hi = grid[b[3 + 8 * sub + 2 * l]];
        return {{signed4(lo, signs), signed4(hi, signs >> 4)}, static_cast<int>(2 * (aux >> 28) + 1), 0};
    }
};

struct Iq4Xs {
    static constexpr unsigned BYTES = 136;
    static constexpr bool MINS = false;
    static __device__ __forceinline__ float min_scale(const uint8_t*) { return 0.0f; }
    static constexpr unsigned TABLE = IQ4NL_OFFSET;
    static __device__ __forceinline__ float scale(const uint8_t* b) { return half_at(b); }
    static __device__ __forceinline__ Lane8 lane(const uint8_t* b, unsigned lane,
                                                 const uint8_t* table) {
        const unsigned sub = lane / 4, p = lane % 4 * 8;
        const unsigned lo = b[4 + sub / 2] >> (4 * (sub % 2)) & 0xf;
        const unsigned hi = (b[2] | b[3] << 8) >> (2 * sub) & 3;
        const uint8_t* codes = b + 8 + 16 * sub + p % 16;
        const unsigned shift = p < 16 ? 0 : 4;
        unsigned w[2];
#pragma unroll
        for (unsigned h = 0; h < 2; ++h) {
            w[h] = 0;
#pragma unroll
            for (unsigned j = 0; j < 4; ++j)
                w[h] |= static_cast<unsigned>(table[codes[4 * h + j] >> shift & 0xf]) << (8 * j);
        }
        return {{static_cast<int>(w[0]), static_cast<int>(w[1])},
                static_cast<int>(lo | hi << 4) - 32, 0};
    }
};

// The 6-bit scale and offset of 32-value group `j` of Q4_K and Q5_K.
__device__ __forceinline__ int2 moe_scale_min(const uint8_t* q, unsigned j) {
    if (j < 4) return make_int2(q[j] & 63, q[j + 4] & 63);
    return make_int2((q[j + 4] & 0xf) | (q[j - 4] >> 6) << 4, (q[j + 4] >> 4) | (q[j] >> 6) << 4);
}

// Each 32 bytes of qs hold 64 values, the low nibbles then the high.
struct Q4K {
    static constexpr unsigned BYTES = 144;
    static constexpr unsigned TABLE = 0;
    static constexpr bool MINS = true;
    static __device__ __forceinline__ float scale(const uint8_t* b) { return half_at(b); }
    static __device__ __forceinline__ float min_scale(const uint8_t* b) { return half_at(b + 2); }
    static __device__ __forceinline__ Lane8 lane(const uint8_t* b, unsigned lane, const uint8_t*) {
        const unsigned n = lane / 8, high = lane % 8 / 4, l = lane % 4 * 8;
        const uint8_t* q = b + 16 + 32 * n + l;
        const int2 sm = moe_scale_min(b + 4, 2 * n + high);
        return {{static_cast<int>(nibbles(q, 4 * high)), static_cast<int>(nibbles(q + 4, 4 * high))},
                sm.x,
                sm.y};
    }
};

// Q4_K with a fifth bit per value from qh, two bits of each byte per 64.
struct Q5K {
    static constexpr unsigned BYTES = 176;
    static constexpr unsigned TABLE = 0;
    static constexpr bool MINS = true;
    static __device__ __forceinline__ float scale(const uint8_t* b) { return half_at(b); }
    static __device__ __forceinline__ float min_scale(const uint8_t* b) { return half_at(b + 2); }
    static __device__ __forceinline__ Lane8 lane(const uint8_t* b, unsigned lane, const uint8_t*) {
        const unsigned n = lane / 8, high = lane % 8 / 4, l = lane % 4 * 8;
        const uint8_t* q = b + 48 + 32 * n + l;
        const uint8_t* qh = b + 16 + l;
        const unsigned bit = 2 * n + high;
        const int2 sm = moe_scale_min(b + 4, bit);
        const unsigned w0 = nibbles(q, 4 * high) | (u32_at(qh) >> bit & 0x01010101u) << 4;
        const unsigned w1 = nibbles(q + 4, 4 * high) | (u32_at(qh + 4) >> bit & 0x01010101u) << 4;
        return {{static_cast<int>(w0), static_cast<int>(w1)}, sm.x, sm.y};
    }
};

// Two halves of 128; in each, ql gives the low 4 bits and qh the top 2 of
// four runs of 32, and values 16 g .. 16 g + 16 have scale g.
struct Q6K {
    static constexpr unsigned BYTES = 210;
    static constexpr unsigned TABLE = 0;
    static constexpr bool MINS = false;
    static __device__ __forceinline__ float scale(const uint8_t* b) { return half_at(b + 208); }
    static __device__ __forceinline__ float min_scale(const uint8_t*) { return 0.0f; }
    static __device__ __forceinline__ Lane8 lane(const uint8_t* b, unsigned lane, const uint8_t*) {
        const unsigned n = lane / 16, k = lane % 16 / 4, l = lane % 4 * 8;
        const uint8_t* ql = b + 64 * n + 32 * (k & 1) + l;
        const uint8_t* qh = b + 128 + 32 * n + l;
        int w[2];
#pragma unroll
        for (unsigned h = 0; h < 2; ++h) {
            const unsigned v =
                nibbles(ql + 4 * h, 4 * (k >> 1)) | (u32_at(qh + 4 * h) >> (2 * k) & 0x03030303u) << 4;
            w[h] = static_cast<int>(__vsub4(v, 0x20202020u));
        }
        return {{w[0], w[1]}, static_cast<int8_t>(b[192 + lane / 2]), 0};
    }
};

// Each 32 bytes of qs hold four 2-bit planes of 32 values; values
// 16 g .. 16 g + 16 have scale and offset g.
struct Q2K {
    static constexpr unsigned BYTES = 84;
    static constexpr unsigned TABLE = 0;
    static constexpr bool MINS = true;
    static __device__ __forceinline__ float scale(const uint8_t* b) { return half_at(b + 80); }
    static __device__ __forceinline__ float min_scale(const uint8_t* b) { return half_at(b + 82); }
    static __device__ __forceinline__ Lane8 lane(const uint8_t* b, unsigned lane, const uint8_t*) {
        const unsigned n = lane / 16, p = lane % 16 / 4, l = lane % 4 * 8;
        const uint8_t* q = b + 16 + 32 * n + l;
        const unsigned s = b[lane / 2];
        return {{static_cast<int>(u32_at(q) >> (2 * p) & 0x03030303u),
                 static_cast<int>(u32_at(q + 4) >> (2 * p) & 0x03030303u)},
                static_cast<int>(s & 0xf),
                static_cast<int>(s >> 4)};
    }
};

// Q2_K's planes with a third bit from hmask, whose clear bits take 4 off.
struct Q3K {
    static constexpr unsigned BYTES = 110;
    static constexpr unsigned TABLE = 0;
    static constexpr bool MINS = false;
    static __device__ __forceinline__ float scale(const uint8_t* b) { return half_at(b + 108); }
    static __device__ __forceinline__ float min_scale(const uint8_t*) { return 0.0f; }
    static __device__ __forceinline__ Lane8 lane(const uint8_t* b, unsigned lane, const uint8_t*) {
        const unsigned n = lane / 16, p = lane % 16 / 4, l = lane % 4 * 8;
        const uint8_t* q = b + 32 + 32 * n + l;
        const uint8_t* hm = b + l;
        const unsigned bit = 4 * n + p;
        int w[2];
#pragma unroll
        for (unsigned h = 0; h < 2; ++h) {
            const unsigned v = (u32_at(q + 4 * h) >> (2 * p) & 0x03030303u) |
                               (u32_at(hm + 4 * h) >> bit & 0x01010101u) << 2;
            w[h] = static_cast<int>(__vsub4(v, 0x04040404u));
        }
        const unsigned g = lane / 2;
        const unsigned low = b[96 + g % 8] >> (4 * (g / 8)) & 0xf;
        const unsigned high = b[104 + g % 4] >> (2 * (g / 4)) & 3;
        return {{w[0], w[1]}, static_cast<int>(low | high << 4) - 32, 0};
    }
};

__device__ __forceinline__ int warp_isum(int v) {
#pragma unroll
    for (unsigned o = WARP / 2; o > 0; o /= 2) v += __shfl_xor_sync(FULL_MASK, v, o);
    return v;
}

// The host's `silu`, operation by operation: a degree-6 polynomial for e^r
// after x = n ln 2 + r, and 2^n from the exponent bits.
__device__ __forceinline__ float exact_exp(float x) {
    if (x > 88.0f) return CUDART_INF_F;
    if (x < -87.0f) return 0.0f;
    const float n = roundf(__fmul_rn(x, 1.442695f));
    const float r = __fsub_rn(__fsub_rn(x, __fmul_rn(n, 0.6933594f)), __fmul_rn(n, -2.1219444e-4f));
    float p = 1.9875691e-4f;
    p = __fadd_rn(__fmul_rn(p, r), 1.3981999e-3f);
    p = __fadd_rn(__fmul_rn(p, r), 8.333452e-3f);
    p = __fadd_rn(__fmul_rn(p, r), 4.1665796e-2f);
    p = __fadd_rn(__fmul_rn(p, r), 0.16666666f);
    p = __fadd_rn(__fmul_rn(p, r), 0.5f);
    const float e = __fadd_rn(__fadd_rn(__fmul_rn(p, __fmul_rn(r, r)), r), 1.0f);
    return __fmul_rn(e, __int_as_float((static_cast<int>(n) + 127) << 23));
}

__device__ __forceinline__ float exact_silu(float x) {
    return __fdiv_rn(x, __fadd_rn(1.0f, exact_exp(-x)));
}

// The host's `soft_clip`, b tanh(x / b), operation by operation: Cephes' odd
// polynomial below 0.625, 1 - 2 / (e^2y + 1) through `exact_exp` above.
__device__ __forceinline__ float exact_soft_clip(float x, float b) {
    const float y = __fdiv_rn(x, b), a = fabsf(y);
    float t;
    if (a < 0.625f) {
        const float z = __fmul_rn(y, y);
        float p = -5.7049887e-3f;
        p = __fadd_rn(__fmul_rn(p, z), 2.063909e-2f);
        p = __fadd_rn(__fmul_rn(p, z), -5.3739716e-2f);
        p = __fadd_rn(__fmul_rn(p, z), 1.3331442e-1f);
        p = __fadd_rn(__fmul_rn(p, z), -3.333328e-1f);
        t = __fadd_rn(__fmul_rn(__fmul_rn(p, z), y), y);
    } else {
        t = __fsub_rn(1.0f, __fdiv_rn(2.0f, __fadd_rn(exact_exp(__fmul_rn(2.0f, a)), 1.0f)));
        if (y < 0.0f) t = -t;
    }
    return __fmul_rn(b, t);
}

struct MoeParams {
    const uint8_t* arena;
    unsigned long long slot_bytes;
    // Per group: its slot, first assignment and number of assignments (at
    // most MOE_GROUP), consecutive.
    const uint4* groups;
    // Per assignment, the input row it reads; null when assignment `a`
    // reads row `a`.
    const unsigned* inputs;
    const int8_t* xq;
    const float* xd;
    float* y;
    const uint8_t* tables;
    // Where the first (gate or down) and second (up) matrix start in a slot.
    unsigned first;
    unsigned second;
    unsigned rows;
    unsigned cols;
    // Gated products take SITU with these betas in place of SiLU.
    unsigned situ;
    float beta;
    float linear;
    // Where a slot's per-expert scales of its gate, up and down products
    // start, or 0 without; the first matrix's is number `scale_index`.
    unsigned scales;
    unsigned scale_index;
    // Most assignments in a group, and rows per block.
    unsigned most;
    unsigned tile;
};

// silu(g) u, or SITU: beta tanh(g / beta) sigmoid(g) times u, itself
// soft-clipped by `linear` unless that is zero.
__device__ __forceinline__ float moe_glu(const MoeParams& p, float g, float u) {
    if (!p.situ) return __fmul_rn(exact_silu(g), u);
    const float up = p.linear == 0.0f ? u : exact_soft_clip(u, p.linear);
    const float sig = __fdiv_rn(1.0f, __fadd_rn(1.0f, exact_exp(-g)));
    return __fmul_rn(__fmul_rn(exact_soft_clip(g, p.beta), sig), up);
}

// A row's value from its sums of the first and (gated) second matrix: each
// times its expert's scale, when the slot carries them, then combined.
template <bool GATED>
__device__ __forceinline__ float moe_output(const MoeParams& p, const uint8_t* slot, float first, float second) {
    if (p.scales) {
        const uint8_t* s = slot + p.scales + 4 * p.scale_index;
        first = __fmul_rn(first, __int_as_float(static_cast<int>(u32_at(s))));
        if (GATED) second = __fmul_rn(second, __int_as_float(static_cast<int>(u32_at(s + 4))));
    }
    return GATED ? moe_glu(p, first, second) : first;
}

// Each product takes two steps. First each thread takes 32 values of a
// row of one matrix and turns their exact integer sums with every
// assignment's input into the terms of their blocks or groups, each times
// its two scales, in shared memory; then one thread per row and assignment
// adds a row's terms in order, as the host does. A block computes
// `p.tile` rows of the expert its group names.

// Stages the inputs of the group's assignments, each with one input scale
// per `per` values read every `stride` floats of `p.xd`, and returns where
// the terms go in shared memory.
__device__ __forceinline__ float* moe_stage(const MoeParams& p, uint4 g, unsigned per, unsigned stride,
                                            int8_t*& xs, float*& ds) {
    extern __shared__ int8_t shared[];
    const unsigned blocks = p.cols / per;
    xs = shared;
    ds = reinterpret_cast<float*>(xs + (p.most * p.cols + 15) / 16 * 16);
    for (unsigned t = 0; t < g.z; ++t) {
        const size_t row = p.inputs ? p.inputs[g.y + t] : g.y + t;
        for (unsigned i = threadIdx.x; i < p.cols / 4; i += blockDim.x)
            reinterpret_cast<int*>(xs + t * p.cols)[i] = reinterpret_cast<const int*>(p.xq + row * p.cols)[i];
        for (unsigned i = threadIdx.x; i < blocks; i += blockDim.x) ds[t * blocks + i] = p.xd[stride * (row * blocks + i)];
    }
    return ds + p.most * blocks;
}

// The second step: per matrix and row, `terms` holds `units` terms of
// `kinds` kinds for each assignment, rows `row_floats` apart; a term of the
// second kind is subtracted.
template <bool GATED>
__device__ __forceinline__ void moe_sum_rows(const MoeParams& p, uint4 g, const float* terms, unsigned units,
                                             unsigned kinds, unsigned row_floats, unsigned row0, unsigned tile) {
    const uint8_t* slot = p.arena + p.slot_bytes * g.x;
    for (unsigned i = threadIdx.x; i < tile * g.z; i += blockDim.x) {
        const unsigned r = i / g.z, t = i % g.z;
        float acc[2] = {0.0f, 0.0f};
#pragma unroll
        for (unsigned m = 0; m < (GATED ? 2 : 1); ++m) {
            const float* row = terms + (m * tile + r) * row_floats + t;
            float a = 0.0f;
            if (kinds == 2) {
                for (unsigned u = 0; u < units; ++u)
                    a = __fsub_rn(__fadd_rn(a, row[2 * u * p.most]), row[(2 * u + 1) * p.most]);
            } else {
#pragma unroll 8
                for (unsigned u = 0; u < units; ++u) a = __fadd_rn(a, row[u * p.most]);
            }
            acc[m] = a;
        }
        p.y[static_cast<size_t>(g.y + t) * p.rows + row0 + r] = moe_output<GATED>(p, slot, acc[0], acc[1]);
    }
}

// The rows of types read per 256 values: eight threads give one block's
// sums.
template <class F, bool GATED>
__device__ void moe_rows(const MoeParams& p) {
    const uint4 g = p.groups[blockIdx.y];
    if (g.z == 0) return;
    int8_t* xs;
    float* ds;
    float* terms = moe_stage(p, g, QK_IQ, 1, xs, ds);
    const unsigned row0 = blockIdx.x * p.tile, tile = min(p.tile, p.rows - row0);
    const unsigned blocks = p.cols / QK_IQ, chunks = p.cols / 32;
    const unsigned row_floats = 2 * blocks * p.most + 1;
    const uint8_t* slot = p.arena + p.slot_bytes * g.x;
    const uint8_t* table = p.tables + F::TABLE;
    const size_t row_bytes = static_cast<size_t>(blocks) * F::BYTES;
    __syncthreads();
    // Each warp takes rows, its lanes 32 values each; every lane takes each
    // step, so that the eight lanes of a block can add up their sums.
    const unsigned lane_id = threadIdx.x % WARP;
    for (unsigned mr = threadIdx.x / WARP; mr < (GATED ? 2 : 1) * tile; mr += blockDim.x / WARP) {
        const unsigned m = mr / tile, r = mr % tile;
        const uint8_t* w = slot + (m == 0 ? p.first : p.second) + (row0 + r) * row_bytes;
        float* row = terms + mr * row_floats;
        for (unsigned c0 = 0; c0 < chunks; c0 += WARP) {
            const unsigned c = c0 + lane_id, b = c / 8;
            const bool live = c < chunks;
            const uint8_t* blk = w + b * F::BYTES;
            int sum[MOE_GROUP] = {}, offsets[MOE_GROUP] = {};
            if (live) {
#pragma unroll
                for (unsigned k = 0; k < 4; ++k) {
                    const unsigned lane = 4 * (c % 8) + k;
                    const Lane8 l = F::lane(blk, lane, table);
#pragma unroll
                    for (unsigned t = 0; t < MOE_GROUP; ++t) {
                        if (t >= g.z) break;
                        const int* x = reinterpret_cast<const int*>(xs + t * p.cols + b * QK_IQ + 8 * lane);
                        sum[t] += l.mul * dp4a(l.w[1], x[1], dp4a(l.w[0], x[0], 0));
                        if (F::MINS) offsets[t] += l.min * dp4a(0x01010101, x[1], dp4a(0x01010101, x[0], 0));
                    }
                }
            }
#pragma unroll
            for (unsigned t = 0; t < MOE_GROUP; ++t) {
                if (t >= g.z) break;
#pragma unroll
                for (unsigned o = 1; o < 8; o *= 2) {
                    sum[t] += __shfl_xor_sync(FULL_MASK, sum[t], o);
                    if (F::MINS) offsets[t] += __shfl_xor_sync(FULL_MASK, offsets[t], o);
                }
            }
            if (!live || c % 8 != 0) continue;
            const float wd = F::scale(blk), wmin = F::min_scale(blk);
            float* out = row + 2 * b * p.most;
            for (unsigned t = 0; t < g.z; ++t) {
                const float xd = ds[t * blocks + b];
                out[t] = __fmul_rn(__fmul_rn(wd, xd), static_cast<float>(sum[t]));
                out[p.most + t] =
                    F::MINS && wmin != 0.0f ? __fmul_rn(__fmul_rn(wmin, xd), static_cast<float>(offsets[t])) : 0.0f;
            }
        }
    }
    __syncthreads();
    moe_sum_rows<GATED>(p, g, terms, blocks, 2, row_floats, row0, tile);
}

// The half-value scale of an NVFP4 group from its unsigned E4M3 byte, as
// kinetic-quant's `ue4m3_half` reads it.
__device__ __forceinline__ float ue4m3_half(unsigned b) {
    if (b == 0x7F) return 0.0f;
    const unsigned m = b & 0x7F;
    const float v = m < 8 ? __fmul_rn(static_cast<float>(m), 1.0f / 512.0f) : __int_as_float((m + (120u << 3)) << 20);
    return __fmul_rn(v, 0.5f);
}

// Four table entries for the nibbles of `q` at `shift`, packed as bytes.
__device__ __forceinline__ int table4(const uint8_t* q, unsigned shift, const uint8_t* table) {
    unsigned v = 0;
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) v |= static_cast<unsigned>(table[q[j] >> shift & 0xf]) << (8 * j);
    return static_cast<int>(v);
}

// Types whose scales each cover GROUP values, 16 or 32, read with inputs
// quantized per 32 as GEMV inputs are: an int8 copy and (d, d * sum) per
// 32. `lane` gives weights v .. v + 8 of a row as signed bytes and returns
// the scale of their group; a block of BLOCK values takes BYTES bytes.
struct Iq4Nl {
    static constexpr unsigned GROUP = 32, BLOCK = 32, BYTES = 18;
    // The low nibbles of the 16 bytes, then the high ones.
    static __device__ __forceinline__ float lane(const uint8_t* row, unsigned v, const uint8_t* tables, int w[2]) {
        const uint8_t* b = row + v / BLOCK * BYTES;
        const unsigned part = v % BLOCK / 8;
        const uint8_t* q = b + 2 + 8 * (part % 2);
        const unsigned shift = part < 2 ? 0 : 4;
        w[0] = table4(q, shift, tables + IQ4NL_OFFSET);
        w[1] = table4(q + 4, shift, tables + IQ4NL_OFFSET);
        return half_at(b);
    }
};

// Four scale bytes, then per group of 16 eight bytes of E2M1 codes, low
// nibbles first.
struct Nvfp4 {
    static constexpr unsigned GROUP = 16, BLOCK = 64, BYTES = 36;
    static __device__ __forceinline__ float lane(const uint8_t* row, unsigned v, const uint8_t* tables, int w[2]) {
        const uint8_t* b = row + v / BLOCK * BYTES;
        const unsigned at = v % BLOCK, group = at / GROUP;
        const uint8_t* q = b + 4 + 8 * group;
        const unsigned shift = at % GROUP < 8 ? 0 : 4;
        w[0] = table4(q, shift, tables + MXFP4_OFFSET);
        w[1] = table4(q + 4, shift, tables + MXFP4_OFFSET);
        return ue4m3_half(b[group]);
    }
};

struct Q8Zero {
    static constexpr unsigned GROUP = 32, BLOCK = 32, BYTES = 34;
    static __device__ __forceinline__ float lane(const uint8_t* row, unsigned v, const uint8_t*, int w[2]) {
        const uint8_t* b = row + v / BLOCK * BYTES;
        const uint8_t* q = b + 2 + v % BLOCK;
        w[0] = static_cast<int>(u32_at(q));
        w[1] = static_cast<int>(u32_at(q + 4));
        return half_at(b);
    }
};

// IQ2_S: per 256 values an f16 d, 32 low index bytes, 32 sign bytes, 8
// bytes of high index bits and 8 of 4-bit scales; each 8 values are a grid
// row with signs, each 16 share the scale d (0.5 + s) / 4.
struct Iq2S {
    static constexpr unsigned GROUP = 16, BLOCK = 256, BYTES = 82;
    static __device__ __forceinline__ float lane(const uint8_t* row, unsigned v, const uint8_t* tables, int w[2]) {
        const uint8_t* b = row + v / BLOCK * BYTES;
        const unsigned k = v % BLOCK / 8;
        const unsigned index = b[2 + k] | (b[66 + k / 4] >> (2 * (k % 4)) & 3) << 8;
        const uint2 g = reinterpret_cast<const uint2*>(tables + IQ2S_OFFSET)[index];
        const unsigned signs = b[34 + k];
        w[0] = signed4(g.x, signs);
        w[1] = signed4(g.y, signs >> 4);
        const unsigned s = b[74 + k / 4] >> (4 * (k % 4 / 2)) & 0xf;
        return __fmul_rn(__fmul_rn(half_at(b), 0.5f + static_cast<float>(s)), 0.25f);
    }
};

// IQ1_S: per 256 values an f16 d, 32 low index bytes and per 32 values a
// u16 of high index bits, a 3-bit scale and the sign of a delta of 1/8.
// Each value is (d (2 s + 1) / 8) (8 g +- 1) for its grid entry g in -1..1.
struct Iq1S {
    static constexpr unsigned GROUP = 32, BLOCK = 256, BYTES = 50;
    static __device__ __forceinline__ float lane(const uint8_t* row, unsigned v, const uint8_t* tables, int w[2]) {
        const uint8_t* b = row + v / BLOCK * BYTES;
        const unsigned sub = v % BLOCK / 32, l = v % 32 / 8;
        const unsigned qh = b[34 + 2 * sub] | b[35 + 2 * sub] << 8;
        const unsigned index = b[2 + 4 * sub + l] | (qh >> (3 * l) & 7) << 8;
        const int8_t* g = reinterpret_cast<const int8_t*>(tables + IQ1S_OFFSET) + 8 * index;
        const int delta = qh & 0x8000 ? -1 : 1;
#pragma unroll
        for (unsigned h = 0; h < 2; ++h) {
            unsigned packed = 0;
#pragma unroll
            for (unsigned j = 0; j < 4; ++j)
                packed |= static_cast<unsigned>(static_cast<uint8_t>(8 * g[4 * h + j] + delta)) << (8 * j);
            w[h] = static_cast<int>(packed);
        }
        const float scale = static_cast<float>(2 * (qh >> 12 & 7) + 1);
        return __fmul_rn(__fmul_rn(half_at(b), scale), 0.125f);
    }
};

// Q2_0: per 64 values an f16 d and 2-bit codes c, each value (c - 1) d.
struct Q2Zero {
    static constexpr unsigned GROUP = 32, BLOCK = 64, BYTES = 18;
    static __device__ __forceinline__ float lane(const uint8_t* row, unsigned v, const uint8_t*, int w[2]) {
        const uint8_t* b = row + v / BLOCK * BYTES;
        const uint8_t* q = b + 2 + v % BLOCK / 4;
#pragma unroll
        for (unsigned h = 0; h < 2; ++h) {
            unsigned packed = 0;
#pragma unroll
            for (unsigned j = 0; j < 4; ++j)
                packed |= static_cast<unsigned>(static_cast<uint8_t>(static_cast<int>(q[h] >> (2 * j) & 3) - 1)) << (8 * j);
            w[h] = static_cast<int>(packed);
        }
        return half_at(b);
    }
};

// `moe_rows` for those types: a thread's 32 values are one group or two,
// each with its own term.
template <class F, bool GATED>
__device__ void moe_rows_grouped(const MoeParams& p) {
    const uint4 g = p.groups[blockIdx.y];
    if (g.z == 0) return;
    int8_t* xs;
    float* ds;
    float* terms = moe_stage(p, g, 32, 2, xs, ds);
    constexpr unsigned PER = 32 / F::GROUP;
    const unsigned row0 = blockIdx.x * p.tile, tile = min(p.tile, p.rows - row0);
    const unsigned chunks = p.cols / 32, units = PER * chunks;
    const unsigned row_floats = units * p.most + 1;
    const uint8_t* slot = p.arena + p.slot_bytes * g.x;
    const size_t row_bytes = static_cast<size_t>(p.cols / F::BLOCK) * F::BYTES;
    __syncthreads();
    // Each warp takes rows, its lanes 32 values each.
    for (unsigned mr = threadIdx.x / WARP; mr < (GATED ? 2 : 1) * tile; mr += blockDim.x / WARP) {
        const unsigned m = mr / tile, r = mr % tile;
        const uint8_t* w = slot + (m == 0 ? p.first : p.second) + (row0 + r) * row_bytes;
        float* row = terms + mr * row_floats;
        for (unsigned c = threadIdx.x % WARP; c < chunks; c += WARP) {
            int sum[PER][MOE_GROUP] = {};
            float wd[PER];
#pragma unroll
            for (unsigned k = 0; k < 4; ++k) {
                const unsigned v = 32 * c + 8 * k, u = k * PER / 4;
                int wq[2];
                wd[u] = F::lane(w, v, p.tables, wq);
#pragma unroll
                for (unsigned t = 0; t < MOE_GROUP; ++t) {
                    if (t >= g.z) break;
                    const int* x = reinterpret_cast<const int*>(xs + t * p.cols + v);
                    sum[u][t] += dp4a(wq[1], x[1], dp4a(wq[0], x[0], 0));
                }
            }
            float* out = row + PER * c * p.most;
            for (unsigned t = 0; t < g.z; ++t) {
                const float xd = ds[t * chunks + c];
#pragma unroll
                for (unsigned u = 0; u < PER; ++u)
                    out[u * p.most + t] = __fmul_rn(__fmul_rn(wd[u], xd), static_cast<float>(sum[u][t]));
            }
        }
    }
    __syncthreads();
    moe_sum_rows<GATED>(p, g, terms, units, 1, row_floats, row0, tile);
}

#define MOE_KERNELS(name, F, ROWS)                                                           \
    extern "C" __global__ void __launch_bounds__(MOE_THREADS) moe_gate_up_##name(MoeParams p) { \
        ROWS<F, true>(p);                                                                    \
    }                                                                                        \
    extern "C" __global__ void __launch_bounds__(MOE_THREADS) moe_down_##name(MoeParams p) {    \
        ROWS<F, false>(p);                                                                   \
    }

MOE_KERNELS(iq2_xxs, Iq2Xxs, moe_rows)
MOE_KERNELS(iq2_xs, Iq2Xs, moe_rows)
MOE_KERNELS(iq3_xxs, Iq3Xxs, moe_rows)
MOE_KERNELS(iq4_xs, Iq4Xs, moe_rows)
MOE_KERNELS(q2_k, Q2K, moe_rows)
MOE_KERNELS(q3_k, Q3K, moe_rows)
MOE_KERNELS(q4_k, Q4K, moe_rows)
MOE_KERNELS(q5_k, Q5K, moe_rows)
MOE_KERNELS(q6_k, Q6K, moe_rows)
MOE_KERNELS(iq4_nl, Iq4Nl, moe_rows_grouped)
MOE_KERNELS(nvfp4, Nvfp4, moe_rows_grouped)
MOE_KERNELS(q8_0, Q8Zero, moe_rows_grouped)
MOE_KERNELS(iq2_s, Iq2S, moe_rows_grouped)
MOE_KERNELS(iq1_s, Iq1S, moe_rows_grouped)
MOE_KERNELS(q2_0, Q2Zero, moe_rows_grouped)

// Routing on the device, for layers whose experts all have a slot.

// Most experts a layer may have for `moe_route`: 32 per lane.
constexpr unsigned ROUTE_MAX_EXPERTS = 32 * WARP;

struct RouteParams {
    // Router logits, `stride` per token, the experts' first.
    const float* gates;
    // Per token and pick, the expert and its weight.
    unsigned* experts;
    float* weights;
    unsigned tokens;
    unsigned n_exp;
    unsigned k;
    unsigned stride;
    // Sigmoid probabilities instead of a softmax.
    unsigned sigmoid;
    // Added to each expert's probability to choose, not to weigh, or null.
    const float* bias;
    // Multiplies the weights once they are shares.
    float scale;
};

// Per token, the softmax (or sigmoid) of the logits, then the `k` likeliest experts,
// largest first and the lowest index among equals, each weighted by its
// share of the `k` probabilities, the sum floored at the smallest normal
// f16 as on the host. One warp per token; every sum is taken in a fixed
// order, so a token's picks never depend on its batch.
extern "C" __global__ void moe_route(RouteParams p) {
    const unsigned t = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (t >= p.tokens) return;
    const unsigned lane = threadIdx.x % WARP;
    const float* g = p.gates + static_cast<size_t>(t) * p.stride;
    float top = -CUDART_INF_F;
    for (unsigned e = lane; e < p.n_exp; e += WARP) top = fmaxf(top, g[e]);
#pragma unroll
    for (unsigned o = WARP / 2; o > 0; o /= 2) top = fmaxf(top, __shfl_xor_sync(FULL_MASK, top, o));
    float sum = 0.0f;
    for (unsigned e = lane; e < p.n_exp; e += WARP) sum = __fadd_rn(sum, expf(__fsub_rn(g[e], top)));
#pragma unroll
    for (unsigned o = WARP / 2; o > 0; o /= 2) sum = __fadd_rn(sum, __shfl_xor_sync(FULL_MASK, sum, o));

    unsigned taken = 0;
    float picked = 0.0f;
    unsigned* experts = p.experts + static_cast<size_t>(t) * p.k;
    float* weights = p.weights + static_cast<size_t>(t) * p.k;
    for (unsigned j = 0; j < p.k; ++j) {
        float best = -CUDART_INF_F, best_prob = 0.0f;
        unsigned at = 0xffffffffu;
        for (unsigned i = 0, e = lane; e < p.n_exp; ++i, e += WARP) {
            const float prob = p.sigmoid ? __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g[e])))
                                         : __fdiv_rn(expf(__fsub_rn(g[e], top)), sum);
            const float score = p.bias ? __fadd_rn(prob, p.bias[e]) : prob;
            if (!(taken >> i & 1u) && score > best) {
                best = score;
                best_prob = prob;
                at = e;
            }
        }
#pragma unroll
        for (unsigned o = WARP / 2; o > 0; o /= 2) {
            const float b = __shfl_xor_sync(FULL_MASK, best, o);
            const float bp = __shfl_xor_sync(FULL_MASK, best_prob, o);
            const unsigned a = __shfl_xor_sync(FULL_MASK, at, o);
            if (b > best || (b == best && a < at)) {
                best = b;
                best_prob = bp;
                at = a;
            }
        }
        if (at % WARP == lane) taken |= 1u << (at / WARP);
        if (lane == 0) {
            experts[j] = at;
            weights[j] = best_prob;
            picked = __fadd_rn(picked, best_prob);
        }
    }
    if (lane == 0) {
        const float total = fmaxf(picked, 6.1035156e-5f);
        for (unsigned j = 0; j < p.k; ++j) weights[j] = __fmul_rn(__fdiv_rn(weights[j], total), p.scale);
    }
}

struct AssignParams {
    // Per token and pick, its expert.
    const unsigned* experts;
    // The cache slot of every expert of the layer.
    const unsigned* slots;
    // Out: groups of up to MOE_GROUP assignments, the rest of `max_groups`
    // empty; per assignment its input row; per pick its assignment.
    uint4* groups;
    unsigned* inputs;
    unsigned* rows;
    unsigned picks;
    unsigned k;
    unsigned n_exp;
    unsigned max_groups;
};

// Gives every pick an assignment, the picks of each expert together in
// pick order and in groups of up to MOE_GROUP, for `moe_rows`. One block;
// shared memory holds the picks and three words per expert.
extern "C" __global__ void moe_assign(AssignParams p) {
    extern __shared__ unsigned assign_shared[];
    unsigned* picked = assign_shared;
    unsigned* count = picked + p.picks;
    unsigned* first = count + p.n_exp;
    unsigned* group = first + p.n_exp;
    __shared__ unsigned total;
    for (unsigned s = threadIdx.x; s < p.picks; s += blockDim.x) picked[s] = p.experts[s];
    __syncthreads();
    for (unsigned e = threadIdx.x; e < p.n_exp; e += blockDim.x) {
        unsigned c = 0;
        for (unsigned s = 0; s < p.picks; ++s) c += picked[s] == e;
        count[e] = c;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned a = 0, n = 0;
        for (unsigned e = 0; e < p.n_exp; ++e) {
            first[e] = a;
            group[e] = n;
            a += count[e];
            n += (count[e] + MOE_GROUP - 1) / MOE_GROUP;
        }
        total = n;
    }
    __syncthreads();
    for (unsigned e = threadIdx.x; e < p.n_exp; e += blockDim.x) {
        unsigned j = 0;
        for (unsigned s = 0; s < p.picks && j < count[e]; ++s) {
            if (picked[s] != e) continue;
            const unsigned a = first[e] + j;
            p.inputs[a] = s / p.k;
            p.rows[s] = a;
            if (j % MOE_GROUP == 0)
                p.groups[group[e] + j / MOE_GROUP] = make_uint4(p.slots[e], a, min(MOE_GROUP, count[e] - j), 0);
            ++j;
        }
    }
    for (unsigned i = total + threadIdx.x; i < p.max_groups; i += blockDim.x) p.groups[i] = make_uint4(0, 0, 0, 0);
}

struct CombineParams {
    // Expert outputs, `d` values per row.
    const float* experts;
    // Per token and pick: the row of `experts` holding its output, and its
    // weight.
    const unsigned* rows;
    const float* weights;
    unsigned picks;
    // Whether `out` holds the shared expert's output; otherwise it starts
    // from zero.
    unsigned shared;
    // The shared expert's output, scaled in place per token by the sigmoid
    // of gates[t * gate_stride] when `gates` is set.
    float* out;
    const float* gates;
    unsigned gate_stride;
    unsigned d;
};

// out = shared * sigmoid(gate) + the weighted outputs of the token's picks,
// added in pick order.
extern "C" __global__ void moe_combine(CombineParams p) {
    const unsigned t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.d) return;
    const size_t at = static_cast<size_t>(t) * p.d + i;
    float v = p.shared ? p.out[at] : 0.0f;
    if (p.gates) v = __fmul_rn(v, sigmoid(p.gates[static_cast<size_t>(t) * p.gate_stride]));
    for (unsigned k = 0; k < p.picks; ++k) {
        const unsigned s = t * p.picks + k;
        v = __fadd_rn(v, __fmul_rn(p.weights[s], p.experts[static_cast<size_t>(p.rows[s]) * p.d + i]));
    }
    p.out[at] = v;
}
