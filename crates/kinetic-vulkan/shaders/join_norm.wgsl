// The input of a low-rank control for each token row of the batch, written
// as a GEMV input of rows `2 d` wide: the row of `x` and then that of `m`,
// each normalized by its root mean square.

struct Shape {
    d: u32,
    eps: f32,
}

var<immediate> shape: Shape;

@group(0) @binding(0) var<storage, read> batch: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> ox: array<f32>;
@group(0) @binding(4) var<storage, read_write> oq: array<u32>;
@group(0) @binding(5) var<storage, read_write> od: array<vec2<f32>>;

const THREADS: u32 = 256u;

var<workgroup> partial: array<f32, 8>;

// The sum of `v` over the workgroup; every invocation calls it together.
fn total(v: f32, slot: u32, lane: u32) -> f32 {
    let s = subgroupAdd(v);
    workgroupBarrier();
    if lane == 0u {
        partial[slot] = s;
    }
    workgroupBarrier();
    var sum = 0.0;
    for (var i = 0u; i < THREADS / 32u; i++) {
        sum += partial[i];
    }
    return sum;
}

fn part(half: u32, i: u32) -> f32 {
    if half == 0u {
        return x[i];
    }
    return m[i];
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
    @builtin(subgroup_id) slot: u32,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let t = group.x;
    if t >= batch[0] {
        return;
    }
    let d = shape.d;
    for (var half = 0u; half < 2u; half++) {
        var ss = 0.0;
        for (var i = tid; i < d; i += THREADS) {
            let v = part(half, t * d + i);
            ss += v * v;
        }
        let scale = 1.0 / sqrt(total(ss, slot, lane) / f32(d) + shape.eps);
        // `d` is a multiple of 32, so every subgroup runs whole.
        for (var i = tid; i < d; i += THREADS) {
            put_q8((2u * t + half) * d + i, part(half, t * d + i) * scale, lane);
        }
    }
}
