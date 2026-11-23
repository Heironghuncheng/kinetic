// Shared part of every expert GEMV kernel; build.rs appends one weight format
// to it, the same as to gemv.wgsl.
//
// The weights are `count` experts of `rows` rows each, one after another as
// one matrix. A pass runs `batch[0] * used` jobs: job j multiplies expert
// `picks[2j]` with input row `j / per_input` and writes output row j from
// `j * stride + offset`; a job whose expert is not below `count` is skipped.
// One subgroup of 32 takes a row of one job, as in gemv.wgsl, and a job's
// sums run in the same order whatever else runs.

struct Shape {
    rows: u32,
    blocks_per_row: u32,
    count: u32,
    used: u32,
    per_input: u32,
    stride: u32,
    offset: u32,
}

var<immediate> shape: Shape;

@group(0) @binding(0) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> xq: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> xd: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read> batch: array<u32>;
// Per job: the expert, then the bits of its weight.
@group(0) @binding(6) var<storage, read> picks: array<u32>;

const ROWS_PER_GROUP: u32 = 4u;
const NIBBLES: u32 = 0x0F0F0F0Fu;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(subgroup_id) slot: u32,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let row = group.x * ROWS_PER_GROUP + slot;
    let job = group.y;
    if row >= shape.rows || job >= batch[0] * shape.used {
        return;
    }
    let expert = picks[2u * job];
    // A workgroup takes one job, so all of it leaves together.
    if expert >= shape.count {
        return;
    }
    let n = shape.blocks_per_row;
    let total = shape.count * shape.rows * n;
    let first = (expert * shape.rows + row) * n;
    let xb = job / shape.per_input * n;
    let part = lane % LANES;
    let step = 32u / LANES;
    var sum = 0.0;
    var b = lane / LANES;
    for (; b + step < n; b += 2u * step) {
        sum += block_dot(total, first + b, xb + b, part)
            + block_dot(total, first + b + step, xb + b + step, part);
    }
    if b < n {
        sum += block_dot(total, first + b, xb + b, part);
    }
    sum = subgroupAdd(sum);
    if lane == 0u {
        y[job * shape.stride + shape.offset + row] = sum;
    }
}

// The two helpers of gemv.wgsl that the weight formats call.
fn dot16(a: vec4<u32>, b: vec4<u32>) -> i32 {
    return dot4I8Packed(a.x, b.x) + dot4I8Packed(a.y, b.y) + dot4I8Packed(a.z, b.z)
        + dot4I8Packed(a.w, b.w);
}

fn f16_at(base: u32, i: u32) -> f32 {
    let pair = unpack2x16float(w[base + i / 8u][(i / 2u) % 4u]);
    return select(pair.x, pair.y, (i & 1u) == 1u);
}
