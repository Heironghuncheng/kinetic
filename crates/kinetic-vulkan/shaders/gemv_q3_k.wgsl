// Q3_K: blocks of 256 weights in groups of 16, stored planar on the device:
// 32 bytes of high bits, 64 bytes of 2-bit planes, 12 bytes of 6-bit scales
// and the f16 scale. Weight `128n + 32p + r` (half `n`, plane `p`) is plane
// `p` of byte `32n + r` of the planes, less 4 where bit `4n + p` of high
// byte `r` is clear, in group `8n + 2p + r / 16`. Eight invocations share a
// block, one per 32 weights, as kinetic-quant reads them.

const LANES: u32 = 8u;

// Scale `g` of the twelve scale bytes at `at`, offset by 32.
fn q3_k_scale(at: u32, g: u32) -> f32 {
    let low = (byte_at(at + g % 8u) >> (4u * (g / 8u))) & 15u;
    let high = (byte_at(at + 8u + g % 4u) >> (2u * (g / 4u))) & 3u;
    return f32(i32(low | (high << 4u)) - 32);
}

fn block_dot(total: u32, blk: u32, b: u32, part: u32) -> f32 {
    let n = part / 4u;
    let plane = part % 4u;
    let hmask = array<vec4<u32>, 2>(w[blk * 2u], w[blk * 2u + 1u]);
    let qs = array<vec4<u32>, 2>(w[total * 2u + blk * 4u + 2u * n], w[total * 2u + blk * 4u + 2u * n + 1u]);
    var p: array<u32, 8>;
    for (var i = 0u; i < 8u; i++) {
        let q = (qs[i / 4u][i % 4u] >> (2u * plane)) & 0x03030303u;
        let h = (hmask[i / 4u][i % 4u] >> (4u * n + plane)) & 0x01010101u;
        p[i] = pack4xI8(vec4<i32>(unpack4xU8(q | (h << 2u))) - vec4<i32>(4));
    }
    let x = b * 16u + 2u * part;
    let lo = dot16(vec4<u32>(p[0], p[1], p[2], p[3]), xq[x]);
    let hi = dot16(vec4<u32>(p[4], p[5], p[6], p[7]), xq[x + 1u]);
    let scales = total * 96u + blk * 12u;
    let g = 2u * part;
    let d = f16_byte(total * 108u + blk * 2u);
    return d * xd[b * 8u + part].x
        * (q3_k_scale(scales, g) * f32(lo) + q3_k_scale(scales, g + 1u) * f32(hi));
}
