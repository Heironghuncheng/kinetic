// Q5_K on the matrix units: Q4_K's nibbles, with bit s of qh[l] the fifth
// bit of element l of sub-block s.

const SUB: u32 = 8u;

// Weights 16h .. 16h + 16 of sub-block `s` of block `blk` to wa[at..].
fn stage(total: u32, blk: u32, s: u32, h: u32, at: u32) {
    let head = w[blk * 11u];
    let qh = w[blk * 11u + 1u + h];
    let q = w[blk * 11u + 3u + 2u * (s / 2u) + h];
    let sm = k4_scale_min(s, head.y, head.z, head.w);
    let dm = unpack2x16float(head.x);
    let a = dm.x * sm.x;
    let b = dm.y * sm.y;
    let shift = 4u * (s % 2u);
    for (var j = 0u; j < 4u; j++) {
        let word = q[j] >> shift;
        let high = qh[j] >> s;
        for (var i = 0u; i < 4u; i++) {
            let v = ((word >> (8u * i)) & 15u) | (((high >> (8u * i)) & 1u) << 4u);
            wa[at + 4u * j + i] = f16(a * f32(v) - b);
        }
    }
}
