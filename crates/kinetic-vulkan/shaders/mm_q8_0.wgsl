// Q8_0 on the matrix units: a block is one step. The device copy is
// planar, all quants then all f16 scales.

const SUB: u32 = 1u;

// Weights 16h .. 16h + 16 of block `blk` to wa[at..].
fn stage(total: u32, blk: u32, s: u32, h: u32, at: u32) {
    let q = w[blk * 2u + h];
    let d = f16(f16_at(total * 2u, blk));
    for (var j = 0u; j < 4u; j++) {
        let v = unpack4xI8(q[j]);
        wa[at + 4u * j] = f16(v.x) * d;
        wa[at + 4u * j + 1u] = f16(v.y) * d;
        wa[at + 4u * j + 2u] = f16(v.z) * d;
        wa[at + 4u * j + 3u] = f16(v.w) * d;
    }
}
