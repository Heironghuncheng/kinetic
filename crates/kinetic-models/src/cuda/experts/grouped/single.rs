//! The products of the parent module for a single input, as each generated
//! token runs them: the integer sums of four or eight groups are reduced
//! together and scaled as one vector, and the scaled sums are then added
//! one group after another, in the scalar form's order.

use std::arch::x86_64::*;
use std::sync::LazyLock;

use kinetic_quant::grids::{ue4m3_half, IQ2S_GRID, MXFP4_VALUES};
use kinetic_quant::iq::IQ4NL_VALUES;

use super::{load16, load32, nibbles, products, signed_products, Input};

/// Every E4M3 scale byte as [`ue4m3_half`] reads it.
static UE4M3: LazyLock<[f32; 256]> = LazyLock::new(|| std::array::from_fn(|b| ue4m3_half(b as u8)));

/// The sums of the low and of the high four lanes of each vector, in the
/// order: first low, first high, second low, and so on.
#[inline(always)]
unsafe fn half_sums(v: [__m256i; 4]) -> __m256i {
    let h = _mm256_hadd_epi32(_mm256_hadd_epi32(v[0], v[1]), _mm256_hadd_epi32(v[2], v[3]));
    _mm256_permutevar8x32_epi32(h, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7))
}

/// The sum of each vector's eight lanes, in order.
#[inline(always)]
unsafe fn sums(v: [__m256i; 8]) -> __m256i {
    let a = _mm256_hadd_epi32(_mm256_hadd_epi32(v[0], v[1]), _mm256_hadd_epi32(v[2], v[3]));
    let b = _mm256_hadd_epi32(_mm256_hadd_epi32(v[4], v[5]), _mm256_hadd_epi32(v[6], v[7]));
    _mm256_add_epi32(
        _mm256_permute2x128_si256::<0x20>(a, b),
        _mm256_permute2x128_si256::<0x31>(a, b),
    )
}

/// `acc` plus `weight * input * sum` of the first `n` groups, added one
/// after another.
#[inline(always)]
unsafe fn add_groups(acc: f32, weight: __m256, input: __m256, sums: __m256i, n: usize) -> f32 {
    let scaled = _mm256_mul_ps(_mm256_mul_ps(weight, input), _mm256_cvtepi32_ps(sums));
    let mut v = [0.0f32; 8];
    // SAFETY: `v` holds eight values.
    unsafe { _mm256_storeu_ps(v.as_mut_ptr(), scaled) };
    v[..n].iter().fold(acc, |acc, &x| acc + x)
}

/// The input scales `ds[at..at + 4]`, each twice.
#[inline(always)]
unsafe fn each_twice(ds: &[f32], at: usize) -> __m256 {
    let ds = &ds[at..at + 4];
    // SAFETY: the slice holds the four values read.
    let x = unsafe { _mm_loadu_ps(ds.as_ptr()) };
    let x = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(x), x);
    _mm256_permutevar8x32_ps(x, _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3))
}

#[target_feature(enable = "avx2")]
pub unsafe fn nvfp4(row: &[u8], (qs, ds): Input) -> f32 {
    let t = unsafe { load16(MXFP4_VALUES.map(|v| v as u8).as_slice()) };
    let scales = &*UE4M3;
    let mut acc = 0.0f32;
    let blocks = row.len() / 36;
    // Two blocks, eight groups of sixteen, at a time; a last odd block
    // fills four of them.
    for b in (0..blocks).step_by(2) {
        let n = (blocks - b).min(2);
        let w = &row[36 * b..36 * (b + n)];
        let mut p = [_mm256_setzero_si256(); 4];
        for (j, p) in p.iter_mut().enumerate().take(2 * n) {
            let (lo, hi) = unsafe { nibbles(load16(&w[36 * (j / 2) + 4 + 16 * (j % 2)..]), t) };
            let v = _mm256_set_m128i(_mm_unpackhi_epi64(lo, hi), _mm_unpacklo_epi64(lo, hi));
            *p = unsafe { products(v, load32(qs, 64 * b + 32 * j)) };
        }
        let d = |i: usize| match i / 4 < n {
            true => scales[w[36 * (i / 4) + i % 4] as usize],
            false => 0.0,
        };
        let weight = _mm256_setr_ps(d(0), d(1), d(2), d(3), d(4), d(5), d(6), d(7));
        let input = match n {
            2 => unsafe { each_twice(ds, 2 * b) },
            _ => {
                let (a, c) = (ds[2 * b], ds[2 * b + 1]);
                _mm256_setr_ps(a, a, c, c, 0.0, 0.0, 0.0, 0.0)
            }
        };
        acc = unsafe { add_groups(acc, weight, input, half_sums(p), 4 * n) };
    }
    acc
}

#[target_feature(enable = "avx2,f16c")]
pub unsafe fn iq4_nl(row: &[u8], (qs, ds): Input) -> f32 {
    let t = unsafe { load16(IQ4NL_VALUES.map(|v| v as u8).as_slice()) };
    let mut acc = 0.0f32;
    let blocks = row.len() / 18;
    // Eight blocks of 32 at a time, the last time fewer.
    for b in (0..blocks).step_by(8) {
        let n = (blocks - b).min(8);
        let w = &row[18 * b..18 * (b + n)];
        let mut p = [_mm256_setzero_si256(); 8];
        let mut d = [0u16; 8];
        let mut input = [0.0f32; 8];
        for (j, (p, d)) in p.iter_mut().zip(&mut d).enumerate().take(n) {
            let w = &w[18 * j..];
            let (lo, hi) = unsafe { nibbles(load16(&w[2..]), t) };
            *p = unsafe { products(_mm256_set_m128i(hi, lo), load32(qs, 32 * (b + j))) };
            *d = u16::from_le_bytes([w[0], w[1]]);
            input[j] = ds[b + j];
        }
        // SAFETY: both arrays hold eight values.
        let (weight, input) = unsafe {
            (
                _mm256_cvtph_ps(_mm_loadu_si128(d.as_ptr().cast())),
                _mm256_loadu_ps(input.as_ptr()),
            )
        };
        acc = unsafe { add_groups(acc, weight, input, sums(p), n) };
    }
    acc
}

/// For each of 32 values, 1 or -1 as bit `j` of `bits` is clear or set.
#[inline(always)]
unsafe fn sign_bytes(bits: i32) -> __m256i {
    let spread = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3,
    );
    let mask = _mm256_set1_epi64x(0x8040_2010_0804_0201u64 as i64);
    let bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(bits), spread);
    let set = _mm256_cmpeq_epi8(_mm256_and_si256(bytes, mask), mask);
    _mm256_or_si256(set, _mm256_set1_epi8(1))
}

#[target_feature(enable = "avx2")]
pub unsafe fn iq2_s(row: &[u8], (qs, ds): Input) -> f32 {
    let mut acc = 0.0f32;
    for (b, w) in row.chunks_exact(82).enumerate() {
        let d = _mm256_set1_ps(super::half(w));
        // Four chunks of 32, eight groups of sixteen, at a time.
        for c0 in [0, 4] {
            let mut p = [_mm256_setzero_si256(); 4];
            for (c, p) in (c0..c0 + 4).zip(&mut p) {
                let word = |at: usize| i32::from_le_bytes(w[at..at + 4].try_into().unwrap());
                let low = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(word(2 + 4 * c)));
                let high = _mm_and_si128(
                    _mm_srlv_epi32(_mm_set1_epi32(w[66 + c] as i32), _mm_setr_epi32(0, 2, 4, 6)),
                    _mm_set1_epi32(3),
                );
                let index = _mm_or_si128(low, _mm_slli_epi32::<8>(high));
                let mut at = [0u32; 4];
                // SAFETY: the array holds four values.
                unsafe { _mm_storeu_si128(at.as_mut_ptr().cast(), index) };
                let g = at.map(|i| i64::from_le_bytes(IQ2S_GRID[i as usize]));
                let m = _mm256_setr_epi64x(g[0], g[1], g[2], g[3]);
                let s = unsafe { sign_bytes(word(34 + 4 * c)) };
                *p = unsafe { signed_products(m, s, load32(qs, 256 * b + 32 * c)) };
            }
            let nib: [i32; 8] =
                std::array::from_fn(|i| (w[74 + c0 + i / 2] >> (4 * (i % 2)) & 0xf) as i32);
            // SAFETY: the array holds eight values.
            let codes = unsafe { _mm256_loadu_si256(nib.as_ptr().cast()) };
            let weight = _mm256_mul_ps(
                _mm256_mul_ps(
                    d,
                    _mm256_add_ps(_mm256_set1_ps(0.5), _mm256_cvtepi32_ps(codes)),
                ),
                _mm256_set1_ps(0.25),
            );
            let input = unsafe { each_twice(ds, 8 * b + c0) };
            acc = unsafe { add_groups(acc, weight, input, half_sums(p), 8) };
        }
    }
    acc
}
