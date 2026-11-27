// NVFP4: blocks of 64 weights, four unsigned E4M3 scales, one per 16, then
// 32 bytes of E2M1 codes; byte i of a group of 16 holds element i in its low
// nibble and element i + 8 in its high one. Stored planar on the device:
// all code runs, then all scale words. Four invocations share a block, one
// per group of 16; the codes map to integers twice their value and the
// scales to half theirs, as kinetic-quant reads them.

const LANES: u32 = 4u;

// A variable, not a constant, so that it can be indexed at run time.
var<private> E2M1: array<i32, 16> = array<i32, 16>(
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
);

// The integers of the four codes at `shift` in the bytes of `word`.
fn e2m1_values(word: u32, shift: u32) -> u32 {
    return pack4xI8(vec4<i32>(
        E2M1[(word >> shift) & 15u],
        E2M1[(word >> (shift + 8u)) & 15u],
        E2M1[(word >> (shift + 16u)) & 15u],
        E2M1[(word >> (shift + 24u)) & 15u],
    ));
}

// Half the value of an unsigned E4M3 scale; 0x7F reads as zero.
fn ue4m3_half(b: u32) -> f32 {
    if b == 0x7Fu {
        return 0.0;
    }
    let m = b & 0x7Fu;
    if m < 8u {
        return f32(m) / 1024.0;
    }
    return bitcast<f32>((m + 960u) << 20u) * 0.5;
}

fn block_dot(total: u32, blk: u32, b: u32, part: u32) -> f32 {
    let codes = w[blk * 2u + part / 2u];
    var w0 = codes.x;
    var w1 = codes.y;
    if (part & 1u) == 1u {
        w0 = codes.z;
        w1 = codes.w;
    }
    let q = vec4<u32>(
        e2m1_values(w0, 0u),
        e2m1_values(w1, 0u),
        e2m1_values(w0, 4u),
        e2m1_values(w1, 4u),
    );
    let scales = w[total * 2u + blk / 4u][blk % 4u];
    let d = ue4m3_half((scales >> (8u * part)) & 0xFFu);
    return d * xd[b * 2u + part / 2u].x * f32(dot16(q, xq[b * 4u + part]));
}
