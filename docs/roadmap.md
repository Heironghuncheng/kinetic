# Roadmap

What we are working on next, roughly in order. Nothing here has a date;
items move when they are ready, and only after they pass the same exactness
and speed checks as everything else. Ideas and pull requests for any of them
are welcome; open an issue first so we can agree on the approach.

## Landing next

- winget and scoop packages; the manifests are in `packaging/`.

## 0.1.x: small releases

Bug fixes and support for more models, weight types, devices and platforms,
shipped often. What the next one brings is the checklist at the top of the
[README](https://github.com/Veloresearch/kinetic#status). None removes a
command, flag or API field; see [Versioning](https://veloresearch.github.io/kinetic/versioning.html).

## 0.2: every popular model llama.cpp runs

- **More on Vulkan.** Llama, Qwen2.5, Qwen3, LFM3, Qwen3.5 and
  mixture-of-experts models such as Inkling 2 run on AMD, Intel and NVIDIA
  GPUs through `--device vulkan`; the other architectures are next, and
  LFM2.5 MoE matching the CPU there.
- **Layer splits on Vulkan.** Dense models larger than the GPU split their
  layers between GPU and CPU through CUDA today (`--gpu-layers`); Vulkan is
  next.
- **Gemma 4 26B A4B on GPUs.** It runs on the CPU today.
- **More block drafters for RAD-1**, such as DSpark drafters for every model
  that has one, and long drafts for every mixture-of-experts and dense model.
- **Shared passes everywhere.** Convoy runs on the CPU and on CUDA when the
  whole model is on the GPU; layer splits, expert offload and Vulkan are next.
- **More vision models.** Today images work with Qwen3.5 and Gemma 4 and
  their `mmproj` projectors; other projectors are next.
- **More on several GPUs.** Dense models spread their layers over several
  GPUs with `--gpus` today; mixture-of-experts models and splitting every
  layer over two GPUs are next.

## Later

- More architectures, driven by the "model not supported" issues.
- Constrained output (`--grammar`, `--json-schema`) in `kinetic run`; it is
  in the server already.
- Split GGUF files (`-00001-of-0000N.gguf`) read directly.
- `n` greater than 1 in the server API.
- `--imatrix` for `kinetic quantize`.
- Tools to convert and sign `.ktc` files from the command line.

Kinetic supports Windows and Linux on x86-64 and, since 0.1.1, macOS on
Apple Silicon, where the GPU runs through Metal. Intel Macs are not planned.
