// Comes first in every kernel built for devices whose subgroups are 64 wide.
// There each half of a subgroup does the work of one subgroup of 32, and
// build.rs turns every subgroup reduction into a call to the function here
// for its operation and type, which reduces over the caller's half alone.
// Lanes 16, 8, 4, 2 and 1 apart combine in turn, so every lane of a half
// ends with the same bits.

fn half_add_f32(v: f32) -> f32 {
    var s = v;
    for (var m = 16u; m > 0u; m /= 2u) {
        s += subgroupShuffleXor(s, m);
    }
    return s;
}

fn half_add_i32(v: i32) -> i32 {
    var s = v;
    for (var m = 16u; m > 0u; m /= 2u) {
        s += subgroupShuffleXor(s, m);
    }
    return s;
}

fn half_add_u32(v: u32) -> u32 {
    var s = v;
    for (var m = 16u; m > 0u; m /= 2u) {
        s += subgroupShuffleXor(s, m);
    }
    return s;
}

fn half_max_f32(v: f32) -> f32 {
    var s = v;
    for (var m = 16u; m > 0u; m /= 2u) {
        s = max(s, subgroupShuffleXor(s, m));
    }
    return s;
}

fn half_min_u32(v: u32) -> u32 {
    var s = v;
    for (var m = 16u; m > 0u; m /= 2u) {
        s = min(s, subgroupShuffleXor(s, m));
    }
    return s;
}
