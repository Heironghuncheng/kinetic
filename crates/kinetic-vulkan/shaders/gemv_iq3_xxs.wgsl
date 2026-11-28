// IQ3_XXS: blocks of 256 weights in groups of 32, stored planar on the
// device: all 64-byte runs of codebook indices, then all 32-byte runs of
// group words, then all f16 scales, then the codebook of 256 words of four
// magnitudes. A group word holds four 7-bit sign indices, one per eight
// weights, and a 4-bit scale on top. Eight invocations share a block, one
// per group, as kinetic-quant reads them.

const LANES: u32 = 8u;

fn block_dot(total: u32, blk: u32, b: u32, part: u32) -> f32 {
    let qs = w[blk * 4u + part / 2u];
    let odd = (part & 1u) == 1u;
    let q = vec2<u32>(select(qs.x, qs.z, odd), select(qs.y, qs.w, odd));
    let a = word_at(total * 64u + blk * 32u + part * 4u);
    let book = (total * 98u + 15u) / 16u * 16u;
    var p: array<u32, 8>;
    for (var l = 0u; l < 4u; l++) {
        let pair = q[l / 2u] >> (16u * (l % 2u));
        let signs = iq_signs(a >> (7u * l));
        p[2u * l] = signed4(word_at(book + 4u * (pair & 255u)), signs, 0u);
        p[2u * l + 1u] = signed4(word_at(book + 4u * ((pair >> 8u) & 255u)), signs, 4u);
    }
    let x = b * 16u + 2u * part;
    let sum = dot16(vec4<u32>(p[0], p[1], p[2], p[3]), xq[x])
        + dot16(vec4<u32>(p[4], p[5], p[6], p[7]), xq[x + 1u]);
    let d = f16_at(total * 6u, blk) * (0.5 + f32(a >> 28u)) * 0.5;
    return d * xd[b * 8u + part].x * f32(sum);
}
