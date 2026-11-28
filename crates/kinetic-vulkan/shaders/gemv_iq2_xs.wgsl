// IQ2_XS: blocks of 256 weights in groups of eight, stored planar on the
// device: 32 index words of 16 bits, 8 bytes of scales, the f16 scale,
// then the codebook of 512 entries of eight magnitudes, two words each. An
// index word holds a 9-bit codebook index and a 7-bit sign index; each 16
// weights have a 4-bit scale `s` giving `d * (0.5 + s) / 4`. Eight
// invocations share a block, one per 32 weights, as kinetic-quant reads
// them.

const LANES: u32 = 8u;

fn block_dot(total: u32, blk: u32, b: u32, part: u32) -> f32 {
    let book = (total * 74u + 15u) / 16u * 16u;
    let run = w[blk * 4u + part / 2u];
    let odd = (part & 1u) == 1u;
    let qs = vec2<u32>(select(run.x, run.z, odd), select(run.y, run.w, odd));
    var p: array<u32, 8>;
    for (var l = 0u; l < 4u; l++) {
        let q = (qs[l / 2u] >> (16u * (l % 2u))) & 0xFFFFu;
        let signs = iq_signs(q >> 9u);
        let index = q & 511u;
        p[2u * l] = signed4(word_at(book + 8u * index), signs, 0u);
        p[2u * l + 1u] = signed4(word_at(book + 8u * index + 4u), signs, 4u);
    }
    let x = b * 16u + 2u * part;
    let lo = dot16(vec4<u32>(p[0], p[1], p[2], p[3]), xq[x]);
    let hi = dot16(vec4<u32>(p[4], p[5], p[6], p[7]), xq[x + 1u]);
    let scales = byte_at(total * 64u + blk * 8u + part);
    let d = f16_byte(total * 72u + blk * 2u) * 0.25;
    let s = vec2<f32>(f32(scales & 15u), f32(scales >> 4u)) + 0.5;
    return d * xd[b * 8u + part].x * (s.x * f32(lo) + s.y * f32(hi));
}
