// A low-rank control for each token row of the batch, in place:
// x = (x - tanh(forget) x) + tanh(gain) m.

struct Shape {
    d: u32,
}

var<immediate> shape: Shape;

@group(0) @binding(0) var<storage, read> batch: array<u32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read> m: array<f32>;
@group(0) @binding(3) var<storage, read> gain: array<f32>;
@group(0) @binding(4) var<storage, read> forget: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let i = group.x * 256u + tid;
    let t = group.y;
    if t >= batch[0] || i >= shape.d {
        return;
    }
    let at = t * shape.d + i;
    let v = x[at];
    x[at] = (v - tanh(forget[at]) * v) + tanh(gain[at]) * m[at];
}
