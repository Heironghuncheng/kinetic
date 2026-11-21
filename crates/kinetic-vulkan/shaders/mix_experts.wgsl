// A mixture-of-experts layer's output for each token row: the shared expert's
// output already in `y`, scaled by the sigmoid of its gate when `mode` is 2,
// or nothing when `mode` is 0, plus each picked expert's output row times its
// weight, added in pick order.

struct Shape {
    d: u32,
    used: u32,
    mode: u32,
}

var<immediate> shape: Shape;

@group(0) @binding(0) var<storage, read> batch: array<u32>;
@group(0) @binding(1) var<storage, read> picks: array<u32>;
@group(0) @binding(2) var<storage, read> routed: array<f32>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let i = group.x * 256u + tid;
    let t = group.y;
    if t >= batch[0] || i >= shape.d {
        return;
    }
    var s = 0.0;
    if shape.mode != 0u {
        s = y[t * shape.d + i];
    }
    if shape.mode == 2u {
        s *= 1.0 / (1.0 + exp(-gate[t]));
    }
    for (var k = 0u; k < shape.used; k++) {
        let job = t * shape.used + k;
        s += bitcast<f32>(picks[2u * job + 1u]) * routed[job * shape.d + i];
    }
    y[t * shape.d + i] = s;
}
