//! Compiles the WGSL kernels in `shaders/` to SPIR-V with naga, so building
//! needs no Vulkan SDK. Each kernel is its constants followed by the
//! concatenation of its source parts; `src/kernels.rs` embeds the resulting
//! `<name>.spv` files, and the `<name>_64.spv` built from them for devices
//! whose subgroups are 64 wide.

use std::env;
use std::fs;
use std::path::PathBuf;

use naga::back::spv;
use naga::valid::{Capabilities, ValidationFlags, Validator};

include!("src/matrix_tile.rs");

/// (kernel, constants, source parts in order).
const KERNELS: &[(&str, &str, &[&str])] = &[
    ("gemv_f32", "", &["gemv", "gemv_f32"]),
    ("gemv_f16", "", &["gemv", "gemv_f16"]),
    ("gemv_q8_0", "", &["gemv", "gemv_q8_0"]),
    ("gemv_q4_k", "", &["gemv", "k4", "gemv_q4_k"]),
    ("gemv_q5_k", "", &["gemv", "k4", "gemv_q5_k"]),
    ("gemv_q6_k", "", &["gemv", "gemv_q6_k"]),
    ("gemv_iq4_nl", "", &["gemv", "gemv_iq4_nl"]),
    ("gemv_nvfp4", "", &["gemv", "gemv_nvfp4"]),
    ("gemv_iq3_xxs", "", &["gemv", "iq", "gemv_iq3_xxs"]),
    ("gemv_iq2_xs", "", &["gemv", "iq", "gemv_iq2_xs"]),
    ("gemv_q3_k", "", &["gemv", "iq", "gemv_q3_k"]),
    ("gemv_iq3_s", "", &["gemv", "iq", "gemv_iq3_s"]),
    ("gemv_iq2_s", "", &["gemv", "iq", "gemv_iq2_s"]),
    ("experts_f32", "", &["experts", "gemv_f32"]),
    ("experts_f16", "", &["experts", "gemv_f16"]),
    ("experts_q8_0", "", &["experts", "gemv_q8_0"]),
    ("experts_q4_k", "", &["experts", "k4", "gemv_q4_k"]),
    ("experts_q5_k", "", &["experts", "k4", "gemv_q5_k"]),
    ("experts_q6_k", "", &["experts", "gemv_q6_k"]),
    ("experts_iq4_nl", "", &["experts", "gemv_iq4_nl"]),
    ("experts_nvfp4", "", &["experts", "gemv_nvfp4"]),
    ("experts_iq3_xxs", "", &["experts", "iq", "gemv_iq3_xxs"]),
    ("experts_iq2_xs", "", &["experts", "iq", "gemv_iq2_xs"]),
    ("experts_q3_k", "", &["experts", "iq", "gemv_q3_k"]),
    ("experts_iq3_s", "", &["experts", "iq", "gemv_iq3_s"]),
    ("experts_iq2_s", "", &["experts", "iq", "gemv_iq2_s"]),
    ("route", "", &["route"]),
    ("mix_experts", "", &["mix_experts"]),
    ("quantize_q8", "", &["quantize_q8"]),
    ("read_sum", "", &["read_sum"]),
    ("norm", "", &["norm", "put_q8"]),
    ("swiglu", "", &["swiglu", "put_q8"]),
    ("argmax", "", &["argmax"]),
    ("rope", "", &["rope"]),
    ("relative", "", &["relative"]),
    ("residual_conv", "", &["residual_conv"]),
    ("depth", "", &["depth"]),
    ("join_norm", "", &["join_norm", "put_q8"]),
    ("control", "", &["control"]),
    ("short_conv", "", &["short_conv", "put_q8"]),
    ("delta_conv", "", &["delta_conv"]),
    ("delta_rule", "", &["delta_rule"]),
    ("gated_norm", "", &["gated_norm", "put_q8"]),
    ("attend_64", "const PAIRS: u32 = 1u;", &["attend", "put_q8"]),
    (
        "attend_128",
        "const PAIRS: u32 = 2u;",
        &["attend", "put_q8"],
    ),
    (
        "attend_256",
        "const PAIRS: u32 = 4u;",
        &["attend", "put_q8"],
    ),
];

/// Kernels on the matrix units, for devices with cooperative f16 matrices
/// (`Features::matrix_f16`): f16 arithmetic and cooperative matrices
/// validate, and their modules take the Vulkan memory model. Not built for
/// Metal, whose simdgroup matrices are 8x8. `MATRIX_TYPES` in
/// src/kernels.rs says which weight type each is for.
const MATRIX_KERNELS: &[(&str, &str, &[&str])] = &[
    ("mm_q8_0", "", &["mm", "mm_q8_0"]),
    ("mm_q4_k", "", &["mm", "k4", "mm_q4_k"]),
    ("mm_q5_k", "", &["mm", "k4", "mm_q5_k"]),
    ("mm_q6_k", "", &["mm", "mm_q6_k"]),
];

/// The GEMV kernels written differently for Metal: there an invocation
/// takes a tile of four rows and the block sums are floats, which is faster
/// on a Mac's GPU and slower on the others. A part with a file in
/// `shaders/metal` is read from there. The other GEMV kernels, as the
/// expert kernels, take one row per subgroup on every driver.
#[cfg(feature = "metal")]
const METAL_GEMV: &[&str] = &[
    "gemv_f32",
    "gemv_f16",
    "gemv_q8_0",
    "gemv_q4_k",
    "gemv_q5_k",
    "gemv_q6_k",
];

/// Metal GEMV kernels that take one part more.
#[cfg(feature = "metal")]
const METAL_PARTS: &[(&str, &str)] = &[
    ("gemv_f32", "gemv_dots"),
    ("gemv_f16", "gemv_dots"),
    ("gemv_q8_0", "gemv_dots"),
    ("gemv_q5_k", "gemv_dots"),
];

/// The source of a kernel: its constants, then its parts from `dirs`, the
/// first that has each.
fn source(constants: &str, parts: &[&str], dirs: &[&str]) -> String {
    std::iter::once(constants.to_string())
        .chain(parts.iter().map(|part| {
            let path = dirs
                .iter()
                .map(|dir| format!("{dir}/{part}.wgsl"))
                .find(|path| fs::exists(path).unwrap_or(false))
                .unwrap_or_else(|| panic!("no shader part {part}"));
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"))
        }))
        .collect::<Vec<_>>()
        .join("\n")
}

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rustc-check-cfg=cfg(metal_driver)");
    let metal =
        cfg!(feature = "metal") && env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "macos");
    if metal {
        println!("cargo:rustc-cfg=metal_driver");
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"));
    for &(name, constants, parts) in MATRIX_KERNELS {
        // after the parts, whose `enable` directives come first
        let source = format!(
            "{}\nconst TILE: u32 = {MATRIX_TILE}u;\n",
            source(constants, parts, &["shaders"])
        );
        let (module, info) = parse_with(name, &source, MATRIX_CAPABILITIES);
        fs::write(out.join(format!("{name}.spv")), spirv(name, &module, &info))
            .expect("write SPIR-V");
    }
    for &(name, constants, parts) in KERNELS {
        let vulkan = source(constants, parts, &["shaders"]);
        let (module, info) = parse(name, &vulkan);
        fs::write(out.join(format!("{name}.spv")), spirv(name, &module, &info))
            .expect("write SPIR-V");
        let (module, info) = parse_wide(name, &wide_source(name, &vulkan));
        fs::write(
            out.join(format!("{name}_64.spv")),
            spirv(name, &module, &info),
        )
        .expect("write SPIR-V");
        #[cfg(feature = "metal")]
        if metal {
            let mut parts = parts.to_vec();
            parts.extend(METAL_PARTS.iter().filter(|p| p.0 == name).map(|p| p.1));
            let one_row = name.starts_with("experts")
                || (name.starts_with("gemv") && !METAL_GEMV.contains(&name));
            let dirs: &[&str] = match one_row {
                true => &["shaders"],
                false => &["shaders/metal", "shaders"],
            };
            let source = source(constants, &parts, dirs);
            let (module, info) = parse(name, &source);
            fs::write(out.join(format!("{name}.metal")), msl(name, &module, &info))
                .expect("write Metal source");
        }
    }
}

/// What every kernel may use.
const CAPABILITIES: Capabilities = Capabilities::IMMEDIATES
    .union(Capabilities::SUBGROUP)
    .union(Capabilities::SHADER_FLOAT16_IN_FLOAT32);

/// What the matrix kernels may use besides.
const MATRIX_CAPABILITIES: Capabilities = CAPABILITIES
    .union(Capabilities::SHADER_FLOAT16)
    .union(Capabilities::COOPERATIVE_MATRIX);

fn parse(name: &str, source: &str) -> (naga::Module, naga::valid::ModuleInfo) {
    parse_with(name, source, CAPABILITIES)
}

fn parse_with(
    name: &str,
    source: &str,
    capabilities: Capabilities,
) -> (naga::Module, naga::valid::ModuleInfo) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{name}:\n{}", e.emit_to_string(source)));
    let info = validate(name, source, &module, capabilities);
    (module, info)
}

fn validate(
    name: &str,
    source: &str,
    module: &naga::Module,
    capabilities: Capabilities,
) -> naga::valid::ModuleInfo {
    Validator::new(ValidationFlags::all(), capabilities)
        .validate(module)
        .unwrap_or_else(|e| panic!("{name}:\n{}", e.emit_to_string(source)))
}

/// The source of a kernel for subgroups of 64: the helpers of
/// `subgroup_64.wgsl` first, then the kernel with `slot` and `lane` naming
/// a half of a subgroup and a lane in that half.
fn wide_source(name: &str, source: &str) -> String {
    const SLOT: &str = "@builtin(subgroup_id) slot: u32,";
    const LANE: &str = "@builtin(subgroup_invocation_id) lane: u32,";
    let mut source = source.to_string();
    let mut lets = String::new();
    if source.contains(LANE) {
        source = source.replace(LANE, "@builtin(subgroup_invocation_id) wide_lane: u32,");
        lets += "\n    let lane = wide_lane % 32u;";
        if source.contains(SLOT) {
            source = source.replace(SLOT, "@builtin(subgroup_id) wide_group: u32,");
            lets += "\n    let slot = 2u * wide_group + wide_lane / 32u;";
        }
    }
    assert!(
        source.matches("@builtin(subgroup").count() == lets.matches("let ").count(),
        "{name}: a subgroup builtin is not declared as `slot` and `lane`"
    );
    let body = source
        .find("fn main(")
        .and_then(|at| source[at..].find(") {").map(|end| at + end + 3))
        .unwrap_or_else(|| panic!("{name}: no fn main"));
    source.insert_str(body, &lets);
    let helpers = fs::read_to_string("shaders/subgroup_64.wgsl").expect("read subgroup_64.wgsl");
    format!("{helpers}\n{source}")
}

/// Parses and validates a [`wide_source`], its reductions halved.
fn parse_wide(name: &str, source: &str) -> (naga::Module, naga::valid::ModuleInfo) {
    let (mut module, info) = parse(name, source);
    halve_reductions(name, &mut module, &info);
    let info = validate(name, source, &module, CAPABILITIES);
    (module, info)
}

/// Replaces each subgroup reduction with a call to the helper of
/// `subgroup_64.wgsl` for its operation and type.
fn halve_reductions(name: &str, module: &mut naga::Module, info: &naga::valid::ModuleInfo) {
    use naga::{CollectiveOperation, Expression, ScalarKind, Statement, SubgroupOperation};
    let helpers: Vec<_> = module
        .functions
        .iter()
        .filter_map(|(h, f)| Some((f.name.clone()?, h)))
        .collect();
    let types = &module.types;
    let functions = module.functions.iter_mut().map(|(h, f)| (f, &info[h]));
    let entries = module
        .entry_points
        .iter_mut()
        .enumerate()
        .map(|(i, e)| (&mut e.function, info.get_entry_point(i)));
    for (function, function_info) in functions.chain(entries) {
        let expressions = &mut function.expressions;
        each_statement(&mut function.body, &mut |statement| {
            let Statement::SubgroupCollectiveOperation {
                op,
                collective_op,
                argument,
                result,
            } = *statement
            else {
                return;
            };
            let op = match (op, collective_op) {
                (SubgroupOperation::Add, CollectiveOperation::Reduce) => "add",
                (SubgroupOperation::Max, CollectiveOperation::Reduce) => "max",
                (SubgroupOperation::Min, CollectiveOperation::Reduce) => "min",
                _ => panic!("{name}: no 64-wide form of subgroup {op:?} {collective_op:?}"),
            };
            let ty = match function_info[argument].ty.inner_with(types).scalar() {
                Some(s) if s.width == 4 && s.kind == ScalarKind::Float => "f32",
                Some(s) if s.width == 4 && s.kind == ScalarKind::Sint => "i32",
                Some(s) if s.width == 4 && s.kind == ScalarKind::Uint => "u32",
                other => panic!("{name}: no 64-wide subgroup {op} of {other:?}"),
            };
            let helper = format!("half_{op}_{ty}");
            let function = helpers
                .iter()
                .find(|(n, _)| *n == helper)
                .unwrap_or_else(|| panic!("{name}: subgroup_64.wgsl has no {helper}"))
                .1;
            expressions[result] = Expression::CallResult(function);
            *statement = Statement::Call {
                function,
                arguments: vec![argument],
                result: Some(result),
            };
        });
    }
}

/// Calls `visit` on every statement of `block` that holds no block itself.
fn each_statement(block: &mut naga::Block, visit: &mut impl FnMut(&mut naga::Statement)) {
    use naga::Statement;
    for statement in block.iter_mut() {
        match statement {
            Statement::Block(inner) => each_statement(inner, visit),
            Statement::If { accept, reject, .. } => {
                each_statement(accept, visit);
                each_statement(reject, visit);
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    each_statement(&mut case.body, visit);
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                each_statement(body, visit);
                each_statement(continuing, visit);
            }
            _ => visit(statement),
        }
    }
}

/// The kernel as a Metal shader. Storage buffer `i` of set 0 is Metal
/// buffer `i`; the push constants follow them, then the sizes naga wants
/// for runtime arrays, which `src/metal/commands.rs` binds in that order.
///
/// Metal wants the size of each block of workgroup memory when a kernel is
/// dispatched, so the first line lists them, in the order the entry point
/// takes them, for `src/metal/pipeline.rs` to read back.
#[cfg(feature = "metal")]
fn msl(name: &str, module: &naga::Module, info: &naga::valid::ModuleInfo) -> String {
    use naga::back::msl;
    let bindings = module
        .global_variables
        .iter()
        .filter_map(|(_, v)| v.binding.as_ref())
        .map(|b| b.binding + 1)
        .max()
        .unwrap_or(0);
    let mut resources = msl::EntryPointResources::default();
    for binding in 0..bindings {
        resources.resources.insert(
            naga::ResourceBinding { group: 0, binding },
            msl::BindTarget {
                buffer: Some(binding as u8),
                mutable: true,
                ..Default::default()
            },
        );
    }
    resources.immediates_buffer = Some(bindings as u8);
    resources.sizes_buffer = Some(bindings as u8 + 1);
    let mut options = msl::Options {
        lang_version: (3, 0),
        ..Default::default()
    };
    options.per_entry_point_map.insert("main".into(), resources);
    let source = msl::write_string(module, info, &options, &msl::PipelineOptions::default())
        .unwrap_or_else(|e| panic!("{name}: {e}"))
        .0;
    let mut layouts = naga::proc::Layouter::default();
    layouts
        .update(module.to_ctx())
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let workgroup: Vec<String> = module
        .global_variables
        .iter()
        .filter(|(_, v)| v.space == naga::AddressSpace::WorkGroup)
        // Metal takes the lengths in multiples of 16 bytes.
        .map(|(_, v)| layouts[v.ty].size.next_multiple_of(16).to_string())
        .collect();
    format!("{THREADGROUP_BYTES}{}\n{source}", workgroup.join(" "))
}

/// What starts the line of workgroup memory sizes in a Metal shader.
#[cfg(feature = "metal")]
const THREADGROUP_BYTES: &str = "// threadgroup bytes: ";

fn spirv(name: &str, module: &naga::Module, info: &naga::valid::ModuleInfo) -> Vec<u8> {
    // Vulkan 1.3 consumes SPIR-V 1.6, which has packed int8 dot products in core.
    let options = spv::Options {
        lang_version: (1, 6),
        flags: spv::WriterFlags::empty(),
        emit_int_div_checks: false,
        force_loop_bounding: false,
        // f16 never crosses a stage boundary in a compute kernel
        use_storage_input_output_16: false,
        ..spv::Options::default()
    };
    let words =
        spv::write_vec(module, info, &options, None).unwrap_or_else(|e| panic!("{name}: {e}"));
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}
