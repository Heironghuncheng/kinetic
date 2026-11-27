#pragma once

#include "mma.cuh"
#include "moe.cuh"

// Routed experts on tensor cores, for groups of up to MOE_WIDE assignments
// of one expert, so that each weight is decoded once for all of them.
//
// The terms stay those of `moe_rows` and `moe_rows_grouped`: an exact
// integer sum, now from int8 mma, which is exact in any order, times the
// two scales, and one thread per row and assignment adds a row's terms in
// order from zero. A row's value keeps its bits.
//
// A lane decodes values 8 t .. 8 t + 8 of each 32 of rows g and g + 8, t and
// g its place in its quad and warp, and puts them in k slots 4 t and 16 + 4 t
// of the A fragments; the B fragments take the inputs in the same order.

constexpr unsigned MOE_WIDE = 64;
constexpr unsigned WIDE_WARPS = 4;
constexpr unsigned WIDE_THREADS = WIDE_WARPS * WARP;
constexpr unsigned WIDE_FRAGS = MOE_WIDE / 8;
// Input values staged per step, and the bytes between staged rows, which
// keep a half warp's fragment loads on distinct banks.
constexpr unsigned WIDE_STEP = 128;
constexpr unsigned WIDE_STRIDE = WIDE_STEP + 32;
constexpr unsigned WIDE_CHUNKS = WIDE_STEP / 32;

// A lane's eight weights of one row, the scale of their term and, for the
// types read per 256 values, their integer multiplier.
struct WideWeights {
    int q[2];
    float scale;
    int mul;
};

// The types read per 256 values: one term per block, its integer sum the
// sums of 16 or 32 values (`SPAN`) times their multipliers.
template <class F, unsigned SPAN>
struct WideBlocks {
    static constexpr bool PER_BLOCK = true;
    static constexpr bool HALVES = SPAN == 16;
    static constexpr unsigned IN_BLOCK = QK_IQ;
    static constexpr unsigned IN_STRIDE = 1;
    static __device__ __forceinline__ size_t row_bytes(unsigned cols) {
        return static_cast<size_t>(cols / QK_IQ) * F::BYTES;
    }
    static __device__ __forceinline__ WideWeights decode(const uint8_t* row, unsigned chunk, unsigned t,
                                                         const uint8_t* tables) {
        const uint8_t* blk = row + chunk / 8 * F::BYTES;
        const Lane8 l = F::lane(blk, 4 * (chunk % 8) + t, tables + F::TABLE);
        return {{l.w[0], l.w[1]}, F::scale(blk), l.mul};
    }
};

// The types scaled per 16 or 32 values, one term each.
template <class F>
struct WideGroups {
    static constexpr bool PER_BLOCK = false;
    static constexpr bool HALVES = F::GROUP == 16;
    static constexpr unsigned IN_BLOCK = 32;
    static constexpr unsigned IN_STRIDE = 2;
    static __device__ __forceinline__ size_t row_bytes(unsigned cols) {
        return static_cast<size_t>(cols / F::BLOCK) * F::BYTES;
    }
    static __device__ __forceinline__ WideWeights decode(const uint8_t* row, unsigned chunk, unsigned t,
                                                         const uint8_t* tables) {
        WideWeights w;
        w.scale = F::lane(row, 32 * chunk + 8 * t, tables, w.q);
        w.mul = 0;
        return w;
    }
};

__device__ __forceinline__ void wide_term(float& acc, float wd, float xd, int sum) {
    acc = __fadd_rn(acc, __fmul_rn(__fmul_rn(wd, xd), static_cast<float>(sum)));
}

// A block computes rows of the expert its group names for every assignment
// of the group: 64 rows, or for a gated product 32, two warps taking the
// first matrix and two the second.
template <class P, bool GATED>
__device__ void moe_wide(const MoeParams& p) {
    const uint4 g = p.groups[blockIdx.y];
    if (g.z == 0) return;
    __shared__ __align__(16) int8_t xs[MOE_WIDE * WIDE_STRIDE];
    __shared__ float xds[MOE_WIDE * WIDE_CHUNKS];
    __shared__ unsigned src[MOE_WIDE];
    __shared__ float second[GATED ? 32 * (MOE_WIDE + 1) : 1];

    const unsigned warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const unsigned gid = lane / 4, t = lane % 4, quad = lane & ~3u;
    const unsigned m = GATED ? warp / 2 : 0;
    const unsigned rows_per_block = GATED ? 32 : 64;
    const unsigned local = 16 * (GATED ? warp % 2 : warp);
    const unsigned row_lo = blockIdx.x * rows_per_block + local + gid, row_hi = row_lo + 8;
    const bool live_lo = row_lo < p.rows, live_hi = row_hi < p.rows;
    const unsigned frags = (g.z + 7) / 8;

    const uint8_t* slot = p.arena + p.slot_bytes * g.x;
    const size_t row_bytes = P::row_bytes(p.cols);
    const uint8_t* matrix = slot + (m == 0 ? p.first : p.second);
    const uint8_t* w_lo = matrix + row_lo * row_bytes;
    const uint8_t* w_hi = matrix + row_hi * row_bytes;

    for (unsigned i = threadIdx.x; i < g.z; i += blockDim.x) src[i] = p.inputs ? p.inputs[g.y + i] : g.y + i;

    float acc[WIDE_FRAGS][4];
    int sums[WIDE_FRAGS][4];
#pragma unroll
    for (unsigned f = 0; f < WIDE_FRAGS; ++f)
#pragma unroll
        for (unsigned e = 0; e < 4; ++e) {
            acc[f][e] = 0.0f;
            sums[f][e] = 0;
        }

    const unsigned in_blocks = p.cols / P::IN_BLOCK;
    for (unsigned k0 = 0; k0 < p.cols; k0 += WIDE_STEP) {
        const unsigned chunks = min(WIDE_STEP, p.cols - k0) / 32;
        __syncthreads();
        for (unsigned i = threadIdx.x; i < g.z * chunks * 2; i += blockDim.x) {
            const unsigned tok = i / (2 * chunks), part = i % (2 * chunks);
            const size_t at = static_cast<size_t>(src[tok]) * p.cols + k0 + 16 * part;
            *reinterpret_cast<int4*>(xs + tok * WIDE_STRIDE + 16 * part) = *reinterpret_cast<const int4*>(p.xq + at);
        }
        for (unsigned i = threadIdx.x; i < g.z * chunks; i += blockDim.x) {
            const unsigned tok = i / chunks, c = i % chunks;
            const unsigned block = (k0 + 32 * c) / P::IN_BLOCK;
            xds[tok * WIDE_CHUNKS + c] = p.xd[P::IN_STRIDE * (static_cast<size_t>(src[tok]) * in_blocks + block)];
        }
        __syncthreads();
#pragma unroll
        for (unsigned c = 0; c < WIDE_CHUNKS; ++c) {
            if (c >= chunks) break;
            const unsigned chunk = k0 / 32 + c;
            WideWeights lo = {{0, 0}, 0.0f, 0}, hi = {{0, 0}, 0.0f, 0};
            if (live_lo) lo = P::decode(w_lo, chunk, t, p.tables);
            if (live_hi) hi = P::decode(w_hi, chunk, t, p.tables);
            // With 16-value spans, the first half of each 32 is lanes 0 and 1
            // of a quad, the second lanes 2 and 3.
            const bool first_half = t < 2;
            const int keep = !P::HALVES || first_half ? -1 : 0;
            const int drop = P::HALVES && !first_half ? -1 : 0;
            const float scale_lo[2] = {__shfl_sync(FULL_MASK, lo.scale, quad), __shfl_sync(FULL_MASK, lo.scale, quad | 2)};
            const float scale_hi[2] = {__shfl_sync(FULL_MASK, hi.scale, quad), __shfl_sync(FULL_MASK, hi.scale, quad | 2)};
            const int mul_lo[2] = {__shfl_sync(FULL_MASK, lo.mul, quad), __shfl_sync(FULL_MASK, lo.mul, quad | 2)};
            const int mul_hi[2] = {__shfl_sync(FULL_MASK, hi.mul, quad), __shfl_sync(FULL_MASK, hi.mul, quad | 2)};
            const bool closes = !P::PER_BLOCK || (chunk + 1) % 8 == 0;
#pragma unroll
            for (unsigned f = 0; f < WIDE_FRAGS; ++f) {
                if (f >= frags) break;
                const int2 x = *reinterpret_cast<const int2*>(xs + (8 * f + gid) * WIDE_STRIDE + 32 * c + 8 * t);
                int part[2][4];
                mma_k32_from(part[0], lo.q[0] & keep, hi.q[0] & keep, lo.q[1] & keep, hi.q[1] & keep, x.x, x.y, 0);
                if (P::HALVES)
                    mma_k32_from(part[1], lo.q[0] & drop, hi.q[0] & drop, lo.q[1] & drop, hi.q[1] & drop, x.x, x.y, 0);
#pragma unroll
                for (unsigned e = 0; e < 4; ++e) {
                    const unsigned h = e / 2;
                    const float xd = xds[(8 * f + 2 * t + e % 2) * WIDE_CHUNKS + c];
                    if (P::PER_BLOCK) {
                        const int* mul = h ? mul_hi : mul_lo;
                        sums[f][e] += mul[0] * part[0][e];
                        if (P::HALVES) sums[f][e] += mul[1] * part[1][e];
                        if (closes) {
                            wide_term(acc[f][e], h ? hi.scale : lo.scale, xd, sums[f][e]);
                            sums[f][e] = 0;
                        }
                    } else {
                        const float* scale = h ? scale_hi : scale_lo;
                        wide_term(acc[f][e], scale[0], xd, part[0][e]);
                        if (P::HALVES) wide_term(acc[f][e], scale[1], xd, part[1][e]);
                    }
                }
            }
        }
    }

    if (GATED && m == 1) {
#pragma unroll
        for (unsigned f = 0; f < WIDE_FRAGS; ++f)
#pragma unroll
            for (unsigned e = 0; e < 4; ++e)
                second[(local + gid + 8 * (e / 2)) * (MOE_WIDE + 1) + 8 * f + 2 * t + e % 2] = acc[f][e];
    }
    if (GATED) __syncthreads();
    if (GATED && m == 1) return;
#pragma unroll
    for (unsigned f = 0; f < WIDE_FRAGS; ++f) {
#pragma unroll
        for (unsigned e = 0; e < 4; ++e) {
            const unsigned tok = 8 * f + 2 * t + e % 2, h = e / 2;
            const unsigned row = h ? row_hi : row_lo;
            if (tok >= g.z || row >= p.rows) continue;
            const float up = GATED ? second[(local + gid + 8 * h) * (MOE_WIDE + 1) + tok] : 0.0f;
            p.y[static_cast<size_t>(g.y + tok) * p.rows + row] = moe_output<GATED>(p, slot, acc[f][e], up);
        }
    }
}

#define MOE_WIDE_KERNELS(name, P)                                                                     \
    extern "C" __global__ void __launch_bounds__(WIDE_THREADS) moe_wide_gate_up_##name(MoeParams p) { \
        moe_wide<P, true>(p);                                                                         \
    }                                                                                                 \
    extern "C" __global__ void __launch_bounds__(WIDE_THREADS) moe_wide_down_##name(MoeParams p) {    \
        moe_wide<P, false>(p);                                                                        \
    }

using WideIq2Xxs = WideBlocks<Iq2Xxs, 32>;
using WideIq2Xs = WideBlocks<Iq2Xs, 16>;
using WideIq3Xxs = WideBlocks<Iq3Xxs, 32>;
using WideIq4Xs = WideBlocks<Iq4Xs, 32>;
using WideQ3K = WideBlocks<Q3K, 16>;
using WideQ6K = WideBlocks<Q6K, 16>;

MOE_WIDE_KERNELS(iq2_xxs, WideIq2Xxs)
MOE_WIDE_KERNELS(iq2_xs, WideIq2Xs)
MOE_WIDE_KERNELS(iq3_xxs, WideIq3Xxs)
MOE_WIDE_KERNELS(iq4_xs, WideIq4Xs)
MOE_WIDE_KERNELS(q3_k, WideQ3K)
MOE_WIDE_KERNELS(q6_k, WideQ6K)
MOE_WIDE_KERNELS(iq4_nl, WideGroups<Iq4Nl>)
MOE_WIDE_KERNELS(nvfp4, WideGroups<Nvfp4>)
MOE_WIDE_KERNELS(q8_0, WideGroups<Q8Zero>)
MOE_WIDE_KERNELS(iq2_s, WideGroups<Iq2S>)
MOE_WIDE_KERNELS(iq1_s, WideGroups<Iq1S>)
MOE_WIDE_KERNELS(q2_0, WideGroups<Q2Zero>)
