//! AVX2 forms of `exact::grouped_dot` for several inputs at once: a row's
//! weights are decoded once for all of them, each group's integer sum comes
//! from vector instructions, which is exact in any order, and the groups'
//! scaled sums are added one after another as the scalar form adds them, so
//! every result keeps its bits.

use std::arch::x86_64::*;

use half::f16;
use kinetic_quant::grids::{ue4m3_half, IQ1S_GRID, IQ2S_GRID, MXFP4_VALUES};
use kinetic_quant::iq::IQ4NL_VALUES;
use kinetic_quant::TensorType;

/// An input quantized per 32 values: the values and one scale per block.
pub type Input<'a> = (&'a [i8], &'a [f32]);

/// Most inputs one call takes.
pub const MOST: usize = 8;

/// `out[i]` = a row of a grouped type times `xs[i]`, for at most [`MOST`]
/// inputs.
pub type Dots = fn(&[u8], &[Input], &mut [f32]);

/// The AVX2 products of rows of a grouped type, if this CPU has AVX2 and
/// the type has them.
pub fn dots_fn(ty: TensorType) -> Option<Dots> {
    if !is_x86_feature_detected!("avx2") {
        return None;
    }
    // SAFETY (each closure): AVX2 was just detected.
    match ty {
        TensorType::IQ4NL => Some(|w, x, out| unsafe { iq4_nl(w, x, out) }),
        TensorType::NVFP4 => Some(|w, x, out| unsafe { nvfp4(w, x, out) }),
        TensorType::Q8_0 => Some(|w, x, out| unsafe { q8_0(w, x, out) }),
        TensorType::Q2_0 => Some(|w, x, out| unsafe { q2_0(w, x, out) }),
        TensorType::IQ2S => Some(|w, x, out| unsafe { iq2_s(w, x, out) }),
        TensorType::IQ1S => Some(|w, x, out| unsafe { iq1_s(w, x, out) }),
        _ => None,
    }
}

/// Per sign byte, eight bytes of 1 or -1 for `_mm256_sign_epi8`.
static SIGN_BYTES: [i64; 256] = {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        let mut v = 0u64;
        let mut j = 0;
        while j < 8 {
            let byte: u64 = if i >> j & 1 == 1 { 0xff } else { 0x01 };
            v |= byte << (8 * j);
            j += 1;
        }
        table[i] = v as i64;
        i += 1;
    }
    table
};

fn half(b: &[u8]) -> f32 {
    f16::from_le_bytes([b[0], b[1]]).to_f32()
}

/// The 32 input values at `at`.
///
/// # Safety
///
/// AVX2 must be available and `x` hold 32 values from `at`.
#[inline(always)]
unsafe fn load32(x: &[i8], at: usize) -> __m256i {
    let x = &x[at..at + 32];
    // SAFETY: the slice holds the 32 bytes read.
    unsafe { _mm256_loadu_si256(x.as_ptr().cast()) }
}

/// Eight lanes that sum to the products of 32 unsigned magnitudes, each
/// given the sign in `signs`, with 32 inputs.
#[inline(always)]
unsafe fn signed_products(magnitudes: __m256i, signs: __m256i, x: __m256i) -> __m256i {
    let p = _mm256_maddubs_epi16(magnitudes, _mm256_sign_epi8(x, signs));
    _mm256_madd_epi16(p, _mm256_set1_epi16(1))
}

/// [`signed_products`] of 32 signed values.
#[inline(always)]
unsafe fn products(w: __m256i, x: __m256i) -> __m256i {
    unsafe { signed_products(_mm256_sign_epi8(w, w), w, x) }
}

/// The sums of the low and of the high four lanes.
#[inline(always)]
unsafe fn sum_halves(v: __m256i) -> (i32, i32) {
    let s = _mm256_hadd_epi32(v, v);
    let s = _mm256_hadd_epi32(s, s);
    (_mm256_extract_epi32::<0>(s), _mm256_extract_epi32::<4>(s))
}

#[inline(always)]
unsafe fn sum(v: __m256i) -> i32 {
    let (a, b) = unsafe { sum_halves(v) };
    a + b
}

/// Sixteen 4-bit codes looked up in `table`: low nibbles then high nibbles.
#[inline(always)]
unsafe fn nibbles(bytes: __m128i, table: __m128i) -> (__m128i, __m128i) {
    let mask = _mm_set1_epi8(0x0f);
    let lo = _mm_and_si128(bytes, mask);
    let hi = _mm_and_si128(_mm_srli_epi16::<4>(bytes), mask);
    (_mm_shuffle_epi8(table, lo), _mm_shuffle_epi8(table, hi))
}

#[inline(always)]
unsafe fn load16(bytes: &[u8]) -> __m128i {
    let bytes = &bytes[..16];
    // SAFETY: the slice holds the 16 bytes read.
    unsafe { _mm_loadu_si128(bytes.as_ptr().cast()) }
}

#[target_feature(enable = "avx2")]
unsafe fn iq4_nl(row: &[u8], xs: &[Input], out: &mut [f32]) {
    let t = unsafe { load16(IQ4NL_VALUES.map(|v| v as u8).as_slice()) };
    let mut acc = [0.0f32; MOST];
    for (b, w) in row.chunks_exact(18).enumerate() {
        let (lo, hi) = unsafe { nibbles(load16(&w[2..]), t) };
        let v = _mm256_set_m128i(hi, lo);
        let d = half(w);
        for (acc, (qs, ds)) in acc.iter_mut().zip(xs) {
            let s = unsafe { sum(products(v, load32(qs, 32 * b))) };
            *acc += d * ds[b] * s as f32;
        }
    }
    out.copy_from_slice(&acc[..out.len()]);
}

#[target_feature(enable = "avx2")]
unsafe fn q8_0(row: &[u8], xs: &[Input], out: &mut [f32]) {
    let mut acc = [0.0f32; MOST];
    for (b, w) in row.chunks_exact(34).enumerate() {
        let v = unsafe { _mm256_loadu_si256(w[2..34].as_ptr().cast()) };
        let d = half(w);
        for (acc, (qs, ds)) in acc.iter_mut().zip(xs) {
            let s = unsafe { sum(products(v, load32(qs, 32 * b))) };
            *acc += d * ds[b] * s as f32;
        }
    }
    out.copy_from_slice(&acc[..out.len()]);
}

#[target_feature(enable = "avx2")]
unsafe fn nvfp4(row: &[u8], xs: &[Input], out: &mut [f32]) {
    let t = unsafe { load16(MXFP4_VALUES.map(|v| v as u8).as_slice()) };
    let mut acc = [0.0f32; MOST];
    for (b, w) in row.chunks_exact(36).enumerate() {
        for p in 0..2 {
            // groups 2p and 2p + 1, sixteen values each, low and high lane
            let (lo, hi) = unsafe { nibbles(load16(&w[4 + 16 * p..]), t) };
            let v = _mm256_set_m128i(_mm_unpackhi_epi64(lo, hi), _mm_unpacklo_epi64(lo, hi));
            let (d0, d1) = (ue4m3_half(w[2 * p]), ue4m3_half(w[2 * p + 1]));
            for (acc, (qs, ds)) in acc.iter_mut().zip(xs) {
                let (s0, s1) = unsafe { sum_halves(products(v, load32(qs, 64 * b + 32 * p))) };
                *acc += d0 * ds[2 * b + p] * s0 as f32;
                *acc += d1 * ds[2 * b + p] * s1 as f32;
            }
        }
    }
    out.copy_from_slice(&acc[..out.len()]);
}

#[target_feature(enable = "avx2")]
unsafe fn q2_0(row: &[u8], xs: &[Input], out: &mut [f32]) {
    // A code times 1 or 4, as masked at each place in a byte, to code - 1.
    let codes = [-1i8, 0, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0];
    let spread = _mm256_setr_epi8(
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7,
        7, 7,
    );
    let (low, high) = (
        _mm256_set1_epi32(0x0000_0c03),
        _mm256_set1_epi32(0x0c03_0000),
    );
    let t = _mm256_broadcastsi128_si256(unsafe { load16(codes.map(|v| v as u8).as_slice()) });
    let mut acc = [0.0f32; MOST];
    for (b, w) in row.chunks_exact(18).enumerate() {
        let d = half(w);
        for h in 0..2 {
            let bytes = u64::from_le_bytes(w[2 + 8 * h..10 + 8 * h].try_into().unwrap());
            let rep = _mm256_shuffle_epi8(_mm256_set1_epi64x(bytes as i64), spread);
            let n = _mm256_or_si256(
                _mm256_and_si256(rep, low),
                _mm256_and_si256(_mm256_srli_epi16::<4>(rep), high),
            );
            let v = _mm256_shuffle_epi8(t, n);
            for (acc, (qs, ds)) in acc.iter_mut().zip(xs) {
                let s = unsafe { sum(products(v, load32(qs, 64 * b + 32 * h))) };
                *acc += d * ds[2 * b + h] * s as f32;
            }
        }
    }
    out.copy_from_slice(&acc[..out.len()]);
}

#[target_feature(enable = "avx2")]
unsafe fn iq2_s(row: &[u8], xs: &[Input], out: &mut [f32]) {
    let mut acc = [0.0f32; MOST];
    for (b, w) in row.chunks_exact(82).enumerate() {
        let d = half(w);
        for c in 0..8 {
            let grid: [i64; 4] = std::array::from_fn(|i| {
                let k = 4 * c + i;
                let index = w[2 + k] as usize | (w[66 + k / 4] as usize >> (2 * (k % 4)) & 3) << 8;
                i64::from_le_bytes(IQ2S_GRID[index])
            });
            let sign: [i64; 4] = std::array::from_fn(|i| SIGN_BYTES[w[34 + 4 * c + i] as usize]);
            let m = _mm256_setr_epi64x(grid[0], grid[1], grid[2], grid[3]);
            let s = _mm256_setr_epi64x(sign[0], sign[1], sign[2], sign[3]);
            let scales = w[74 + c];
            let d0 = d * (0.5 + (scales & 0xf) as f32) * 0.25;
            let d1 = d * (0.5 + (scales >> 4) as f32) * 0.25;
            for (acc, (qs, ds)) in acc.iter_mut().zip(xs) {
                let x = unsafe { load32(qs, 256 * b + 32 * c) };
                let (s0, s1) = unsafe { sum_halves(signed_products(m, s, x)) };
                *acc += d0 * ds[8 * b + c] * s0 as f32;
                *acc += d1 * ds[8 * b + c] * s1 as f32;
            }
        }
    }
    out.copy_from_slice(&acc[..out.len()]);
}

#[target_feature(enable = "avx2")]
unsafe fn iq1_s(row: &[u8], xs: &[Input], out: &mut [f32]) {
    let ones = _mm256_set1_epi8(1);
    let mut acc = [0.0f32; MOST];
    for (b, w) in row.chunks_exact(50).enumerate() {
        let d = half(w);
        for sub in 0..8 {
            let qh = u16::from_le_bytes([w[34 + 2 * sub], w[35 + 2 * sub]]) as usize;
            let grid: [i64; 4] = std::array::from_fn(|l| {
                let index = w[2 + 4 * sub + l] as usize | (qh >> (3 * l) & 7) << 8;
                i64::from_le_bytes(IQ1S_GRID[index].map(|v| v as u8))
            });
            let g = _mm256_setr_epi64x(grid[0], grid[1], grid[2], grid[3]);
            let delta = if qh & 0x8000 != 0 { -1 } else { 1 };
            let scale = d * (2 * (qh >> 12 & 7) + 1) as f32 * 0.125;
            for (acc, (qs, ds)) in acc.iter_mut().zip(xs) {
                let x = unsafe { load32(qs, 256 * b + 32 * sub) };
                // values are 8 g + delta
                let (dot, total) = unsafe {
                    (
                        sum(signed_products(ones, g, x)),
                        sum(signed_products(ones, ones, x)),
                    )
                };
                *acc += scale * ds[8 * b + sub] * (8 * dot + delta * total) as f32;
            }
        }
    }
    out.copy_from_slice(&acc[..out.len()]);
}
