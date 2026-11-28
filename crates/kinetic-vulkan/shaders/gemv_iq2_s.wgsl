// IQ2_S: blocks of 256 weights in groups of eight, stored planar on the
// device: 32 bytes of low index bits, 32 sign bytes, 8 bytes of high index
// bits, 8 bytes of scales, the f16 scale, then the codebook of 1024 entries
// of eight magnitudes, two words each. Each 16 weights have a 4-bit scale
// `s` giving `d * (0.5 + s) / 4`. Eight invocations share a block, one per
// 32 weights, as kinetic-quant reads them.

const LANES: u32 = 8u;

fn block_dot(total: u32, blk: u32, b: u32, part: u32) -> f32 {
    let book = (total * 82u + 15u) / 16u * 16u;
    let qh = byte_at(total * 64u + blk * 8u + part);
    let scales = byte_at(total * 72u + blk * 8u + part);
    var p: array<u32, 8>;
    for (var l = 0u; l < 4u; l++) {
        let k = 4u * part + l;
        let index = byte_at(blk * 32u + k) | (((qh >> (2u * l)) & 3u) << 8u);
        let signs = byte_at(total * 32u + blk * 32u + k);
        p[2u * l] = signed4(word_at(book + 8u * index), signs, 0u);
        p[2u * l + 1u] = signed4(word_at(book + 8u * index + 4u), signs, 4u);
    }
    let x = b * 16u + 2u * part;
    let lo = dot16(vec4<u32>(p[0], p[1], p[2], p[3]), xq[x]);
    let hi = dot16(vec4<u32>(p[4], p[5], p[6], p[7]), xq[x + 1u]);
    let d = f16_at(total * 5u, blk) * 0.25;
    let s = vec2<f32>(f32(scales & 15u), f32(scales >> 4u)) + 0.5;
    return d * xd[b * 8u + part].x * (s.x * f32(lo) + s.y * f32(hi));
}
