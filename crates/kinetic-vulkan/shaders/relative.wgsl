// Readies one head of one token for attention by relative distance, in place
// in the token's row of `qkv`. A group below `n_head` takes query head x:
// its queries times their scales, clipped to a norm of at most `qk_clip`,
// and its relative terms times theirs, clipped to `rel_clip`, both then
// times the token's log scale 1 + alpha ln(max(1, (pos + 1) / floor)). The
// next `n_kv` groups clip a key head to `qk_clip`, the last `n_kv` give a
// value head its RMS norm; keys and values are then stored as f16 in the
// caches at the token's position, in slot position % slots.

struct Shape {
    n_head: u32,
    n_kv: u32,
    head_dim: u32,
    n_rel: u32,
    stride: u32,
    rel_at: u32,
    k_at: u32,
    v_at: u32,
    qk_clip: f32,
    rel_clip: f32,
    floor: f32,
    alpha: f32,
    eps: f32,
    slots: u32,
    q_scale: u32,
    rel_scale: u32,
}

var<immediate> shape: Shape;

@group(0) @binding(0) var<storage, read> batch: array<u32>;
@group(0) @binding(1) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> consts: array<f32>;
@group(0) @binding(3) var<storage, read_write> kc: array<u32>;
@group(0) @binding(4) var<storage, read_write> vc: array<u32>;

const NO_SCALE: u32 = 0xFFFFFFFFu;

// Scales `len` values at `base` by the constants at `scales`, if any, then
// down to a norm of at most `most`, then by `tau`.
fn clip(base: u32, len: u32, scales: u32, most: f32, tau: f32, lane: u32) {
    var ss = 0.0;
    for (var e = lane; e < len; e += 32u) {
        var v = qkv[base + e];
        if scales != NO_SCALE {
            v *= consts[scales + e];
            qkv[base + e] = v;
        }
        ss += v * v;
    }
    let norm = sqrt(subgroupAdd(ss)) + 1e-6;
    let f = clamp(most / norm, 0.0, 1.0);
    for (var e = lane; e < len; e += 32u) {
        qkv[base + e] = qkv[base + e] * f * tau;
    }
}

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let t = group.y;
    if t >= batch[0] {
        return;
    }
    let slot = group.x;
    let hd = shape.head_dim;
    let pos = batch[1] + t;
    let row = t * shape.stride;
    if slot < shape.n_head {
        let tau = 1.0 + shape.alpha * log(max(f32(pos + 1u) * (1.0 / shape.floor), 1.0));
        clip(row + slot * hd, hd, shape.q_scale + slot * hd, shape.qk_clip, tau, lane);
        let r = slot * shape.n_rel;
        clip(row + shape.rel_at + r, shape.n_rel, shape.rel_scale + r, shape.rel_clip, tau, lane);
        return;
    }
    let keys = slot < shape.n_head + shape.n_kv;
    let head = select(slot - shape.n_head - shape.n_kv, slot - shape.n_head, keys);
    let base = row + select(shape.v_at, shape.k_at, keys) + head * hd;
    if keys {
        clip(base, hd, NO_SCALE, shape.qk_clip, 1.0, lane);
    } else {
        var ss = 0.0;
        for (var e = lane; e < hd; e += 32u) {
            let v = qkv[base + e];
            ss += v * v;
        }
        let scale = 1.0 / sqrt(subgroupAdd(ss) / f32(hd) + shape.eps);
        for (var e = lane; e < hd; e += 32u) {
            qkv[base + e] = qkv[base + e] * scale;
        }
    }
    storageBarrier();
    let at = (pos % shape.slots * shape.n_kv * hd + head * hd) / 2u;
    for (var i = lane; i < hd / 2u; i += 32u) {
        let word = pack2x16float(vec2<f32>(qkv[base + 2u * i], qkv[base + 2u * i + 1u]));
        if keys {
            kc[at + i] = word;
        } else {
            vc[at + i] = word;
        }
    }
}
