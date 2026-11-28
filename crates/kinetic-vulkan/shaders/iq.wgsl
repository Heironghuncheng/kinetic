// What the codebook formats share: words and bytes of the weights at any
// byte offset that is a multiple of four, and of one, and signs applied to
// magnitudes. build.rs puts it after gemv.wgsl or experts.wgsl.

// The little-endian word at byte `at`, a multiple of four.
fn word_at(at: u32) -> u32 {
    return w[at / 16u][(at / 4u) % 4u];
}

fn byte_at(at: u32) -> u32 {
    return (word_at(at & ~3u) >> (8u * (at & 3u))) & 255u;
}

// The eight sign bits of a 7-bit sign index, with the eighth that makes the
// count of set bits even.
fn iq_signs(index: u32) -> u32 {
    let s = index & 127u;
    return s | ((countOneBits(s) & 1u) << 7u);
}

// The four magnitudes of `g`, negated where bits `at` on of `signs` say, as
// four int8 values.
fn signed4(g: u32, signs: u32, at: u32) -> u32 {
    var out = 0u;
    for (var k = 0u; k < 4u; k++) {
        let m = (g >> (8u * k)) & 255u;
        let v = select(m, (256u - m) & 255u, ((signs >> (at + k)) & 1u) == 1u);
        out |= v << (8u * k);
    }
    return out;
}

// The f16 at byte `at`, a multiple of two.
fn f16_byte(at: u32) -> f32 {
    return unpack2x16float(word_at(at & ~3u) >> (8u * (at & 2u))).x;
}
