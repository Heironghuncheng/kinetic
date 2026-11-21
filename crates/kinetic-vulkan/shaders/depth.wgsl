// Attention over depth for each token row of the batch: the states
// normalized by their root mean square, weighted by the softmax of their dot
// products with `query` over the square root of `d`. The states are the
// `n_stack` rows of `stack`, `stack_stride` values apart, then the row of
// `cur` when `with_cur` is set.

struct Shape {
    d: u32,
    n_stack: u32,
    stack_stride: u32,
    with_cur: u32,
}

var<immediate> shape: Shape;

@group(0) @binding(0) var<storage, read> batch: array<u32>;
@group(0) @binding(1) var<storage, read> stack: array<f32>;
@group(0) @binding(2) var<storage, read> cur: array<f32>;
@group(0) @binding(3) var<storage, read> query: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

const THREADS: u32 = 256u;
const MAX_STATES: u32 = 8u;
// The epsilon of these norms, fixed by the reference.
const EPS: f32 = 1e-6;

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

// Element `e` of token `t`'s row of state `s`.
fn state(s: u32, t: u32, e: u32) -> f32 {
    if s < shape.n_stack {
        return stack[s * shape.stack_stride + t * shape.d + e];
    }
    return cur[t * shape.d + e];
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
    let n = shape.n_stack + shape.with_cur;
    var inv: array<f32, MAX_STATES>;
    for (var s = 0u; s < n; s++) {
        var ss = 0.0;
        for (var e = tid; e < d; e += THREADS) {
            let v = state(s, t, e);
            ss += v * v;
        }
        inv[s] = 1.0 / sqrt(total(ss, slot, lane) / f32(d) + EPS);
    }
    let scale = 1.0 / sqrt(f32(d));
    var p: array<f32, MAX_STATES>;
    var top = -3.4e38;
    for (var s = 0u; s < n; s++) {
        var dot = 0.0;
        for (var e = tid; e < d; e += THREADS) {
            dot += state(s, t, e) * inv[s] * query[e];
        }
        p[s] = total(dot, slot, lane) * scale;
        top = max(top, p[s]);
    }
    var sum = 0.0;
    for (var s = 0u; s < n; s++) {
        p[s] = exp(p[s] - top);
        sum += p[s];
    }
    for (var e = tid; e < d; e += THREADS) {
        var o = 0.0;
        for (var s = 0u; s < n; s++) {
            o += p[s] / sum * (state(s, t, e) * inv[s]);
        }
        out[t * d + e] = o;
    }
}
