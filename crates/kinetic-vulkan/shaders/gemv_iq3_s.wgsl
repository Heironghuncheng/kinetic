// IQ3_S: blocks of 256 weights in groups of four, stored planar on the
// device: 64 bytes of low index bits, 8 bytes of high index bits, 32 sign
// bytes, 4 bytes of scales, the f16 scale, then the codebook of 512 words
// of four magnitudes. Each 32 weights have a 4-bit scale `s` giving
// `d * (1 + 2s)`. Eight invocations share a block, one per 32 weights, as
// kinetic-quant reads them.

const LANES: u32 = 8u;

fn block_dot(total: u32, blk: u32, b: u32, part: u32) -> f32 {
    let book = (total * 110u + 15u) / 16u * 16u;
    let qh = byte_at(total * 64u + blk * 8u + part);
    var p: array<u32, 8>;
    for (var l = 0u; l < 8u; l++) {
        let k = 8u * part + l;
        let index = byte_at(blk * 64u + k) | (((qh >> l) & 1u) << 8u);
        let signs = byte_at(total * 72u + blk * 32u + k / 2u);
        p[l] = signed4(word_at(book + 4u * index), signs, 4u * (l % 2u));
    }
    let x = b * 16u + 2u * part;
    let sum = dot16(vec4<u32>(p[0], p[1], p[2], p[3]), xq[x])
        + dot16(vec4<u32>(p[4], p[5], p[6], p[7]), xq[x + 1u]);
    let scale = (byte_at(total * 104u + blk * 4u + part / 2u) >> (4u * (part % 2u))) & 15u;
    let d = f16_byte(total * 108u + blk * 2u) * f32(1u + 2u * scale);
    return d * xd[b * 8u + part].x * f32(sum);
}
