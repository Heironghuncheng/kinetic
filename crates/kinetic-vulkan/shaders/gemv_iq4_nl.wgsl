// IQ4_NL is stored planar on the device, like Q8_0: all 16-byte runs of
// codes, then all f16 scales. Byte i of a run holds element i in its low
// nibble and element i + 16 in its high one; each code indexes a fixed table
// of int8 values. Two invocations share a block, one per nibble.

const LANES: u32 = 2u;

// A variable, not a constant, so that it can be indexed at run time.
var<private> IQ4_NL: array<i32, 16> = array<i32, 16>(
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
);

// The table values of the four codes at `shift` in the bytes of `word`.
fn iq4_nl_values(word: u32, shift: u32) -> u32 {
    return pack4xI8(vec4<i32>(
        IQ4_NL[(word >> shift) & 15u],
        IQ4_NL[(word >> (shift + 8u)) & 15u],
        IQ4_NL[(word >> (shift + 16u)) & 15u],
        IQ4_NL[(word >> (shift + 24u)) & 15u],
    ));
}

fn block_dot(total: u32, blk: u32, b: u32, part: u32) -> f32 {
    let codes = w[blk];
    let shift = 4u * part;
    let q = vec4<u32>(
        iq4_nl_values(codes.x, shift),
        iq4_nl_values(codes.y, shift),
        iq4_nl_values(codes.z, shift),
        iq4_nl_values(codes.w, shift),
    );
    let xv = xq[b * 2u + part];
    return f16_at(total, blk) * xd[b].x * f32(dot16(q, xv));
}
