// The experts each token of the pass runs: its router logits become
// probabilities, by a softmax or a sigmoid each, and the `used` experts of
// the largest probability plus bias are picked, largest first and the lowest
// index among equals, each weighted by its share of their probabilities. One
// invocation per token, so a token's picks never depend on the others.

struct Shape {
    experts: u32,
    used: u32,
    sigmoid: u32,
}

var<immediate> shape: Shape;

@group(0) @binding(0) var<storage, read> batch: array<u32>;
@group(0) @binding(1) var<storage, read_write> logits: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
// Per pick: the expert, then the bits of its weight.
@group(0) @binding(3) var<storage, read_write> picks: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let t = id.x;
    if t >= batch[0] {
        return;
    }
    let base = t * shape.experts;
    if shape.sigmoid != 0u {
        for (var e = 0u; e < shape.experts; e++) {
            logits[base + e] = 1.0 / (1.0 + exp(-logits[base + e]));
        }
    } else {
        var top = -3.4e38;
        for (var e = 0u; e < shape.experts; e++) {
            top = max(top, logits[base + e]);
        }
        var sum = 0.0;
        for (var e = 0u; e < shape.experts; e++) {
            let p = exp(logits[base + e] - top);
            logits[base + e] = p;
            sum += p;
        }
        for (var e = 0u; e < shape.experts; e++) {
            logits[base + e] /= sum;
        }
    }

    // Each pick is the best expert ranked after the one before it.
    var last_score = 0.0;
    var last = 0u;
    var sum = 0.0;
    for (var s = 0u; s < shape.used; s++) {
        var score = -3.4e38;
        var best = 0u;
        var found = false;
        for (var e = 0u; e < shape.experts; e++) {
            let v = logits[base + e] + bias[e];
            let after = s == 0u || v < last_score || (v == last_score && e > last);
            if after && (!found || v > score) {
                score = v;
                best = e;
                found = true;
            }
        }
        last_score = score;
        last = best;
        let p = logits[base + best];
        sum += p;
        picks[2u * (t * shape.used + s)] = best;
        picks[2u * (t * shape.used + s) + 1u] = bitcast<u32>(p);
    }
    // The floor is the smallest normal f16, as in the reference.
    sum = max(sum, 6.1035156e-5);
    for (var s = 0u; s < shape.used; s++) {
        let at = 2u * (t * shape.used + s) + 1u;
        picks[at] = bitcast<u32>(bitcast<f32>(picks[at]) / sum);
    }
}
