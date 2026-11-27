// Causal short convolution with a residual over each token row of the batch:
// out = (x + c) / sqrt(k + 1), where c is the convolution of the channel's
// last `taps` inputs with its weights, oldest first, and k the number of
// taps. With `dynamic`, each row also holds `taps` weights per head of
// `head_len` channels at `taps_at`, newest input first, which add to c and
// count in k. Inputs are read at `x_at` of rows `in_stride` apart; the
// result goes to `out` at `out_at` of rows `out_stride` apart, added to what
// is there when `accumulate` is set.
//
// Each token stores its input in the ring, at slot position % slots, and
// reads those of earlier positions from it, or from the rows of earlier
// tokens of the same pass, as short_conv.wgsl does.

struct Shape {
    len: u32,
    taps: u32,
    slots: u32,
    in_stride: u32,
    x_at: u32,
    taps_at: u32,
    dynamic: u32,
    head_len: u32,
    out_stride: u32,
    out_at: u32,
    accumulate: u32,
}

var<immediate> shape: Shape;

@group(0) @binding(0) var<storage, read> batch: array<u32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> ring: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

// The input of channel `ch` at position `p`, at most the last of the pass.
fn input(p: u32, ch: u32) -> f32 {
    let pos0 = batch[1];
    if p >= pos0 {
        return src[(p - pos0) * shape.in_stride + shape.x_at + ch];
    }
    return ring[p % shape.slots * shape.len + ch];
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let ch = group.x * 256u + tid;
    let t = group.y;
    if t >= batch[0] || ch >= shape.len {
        return;
    }
    let pos = batch[1] + t;
    let w = ch * shape.taps;
    let own = input(pos, ch);
    var acc = 0.0;
    for (var j = 0u; j + 1u < shape.taps; j++) {
        let back = shape.taps - 1u - j;
        if pos >= back {
            acc += weight[w + j] * input(pos - back, ch);
        }
    }
    acc += weight[w + shape.taps - 1u] * own;
    var k = shape.taps;
    if shape.dynamic != 0u {
        let lags = t * shape.in_stride + shape.taps_at + ch / shape.head_len * shape.taps;
        acc += src[lags] * own;
        for (var lag = 1u; lag < shape.taps; lag++) {
            if pos >= lag {
                acc += src[lags + lag] * input(pos - lag, ch);
            }
        }
        k = 2u * shape.taps;
    }
    let y = (own + acc) * (1.0 / sqrt(f32(k + 1u)));
    ring[pos % shape.slots * shape.len + ch] = own;
    let o = t * shape.out_stride + shape.out_at + ch;
    if shape.accumulate != 0u {
        out[o] += y;
    } else {
        out[o] = y;
    }
}
