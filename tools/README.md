# NInfer tools

`tools/` contains artifact conversion and inspection, benchmark orchestration, and serving smoke
checks. To download and run an existing artifact, start with the [project README](../README.md).
To build your own weights, use the [weight conversion guide](../docs/weight-conversion.md).

Run commands from the repository root with a Python environment containing the dependencies
for the selected tool. The maintained environment on this port is the virtualenv at
`C:\vllm-env` (Python 3.12.13, PyTorch 2.11.0+cu130, NumPy 2.3.5); the commands below are
written as `python3`, so substitute `%PY%` for `C:\vllm-env\Scripts\python.exe`. See
[Which Python](../docs/weight-conversion.md#which-python).

Python tools are independent of CMake; there is no `NINFER_BUILD_TOOLS` option.
Tools that launch native binaries require an explicit executable or build-directory argument.
The stored-scale CUDA quantization extension uses PyTorch's visible-device architecture selection;
`TORCH_CUDA_ARCH_LIST` can set an explicit architecture list.

## Release tools

`release/` holds the gates and the packaging pipeline. The twelve that run on every commit are wired
into `.githooks/pre-commit`, and [the row for that hook in `AGENTS.md`](../AGENTS.md) is the
authority for what they cover:

| Tool | Runs |
|---|---|
| `check_doc_links.py` | every commit, and again during packaging |
| `check_doc_citations.py` | every commit; a citation must still point inside the file it names |
| `check_text_encoding.py` | every commit |
| `check_fp8_band_ladders.py` | every commit; its `--self-test` is the control |
| `check_profile_consistency.py` | every commit, and again during packaging |
| `check_calibration_corpus.py` | every commit |
| `check_production_stream_defaults.py` | every commit |
| `check_dead_types.py` | every commit; a declared first-party type nothing names again |
| `check_duplication.py` | every commit; copy-paste added past the recorded baseline |
| `check_port_delta.py --check` | every commit; divergence inside an upstream-owned file past `port_delta_baseline.json` |
The rest run when their stage calls for them:

| Tool | Runs |
|---|---|
| `check_test_baseline.py` | after a C++ or upstream change, with `NINFER_TEST_ARTIFACT` set |
| `check_test_mutation.py` | when an Op needs its acceptance argued, named in `src/ops/CMakeLists.txt` |
| `check_cache_capacity.py` | during packaging |
| `check_archive_contents.py` | on a built archive; verifies every entry against its `SHA256SUMS` |
| `check_release_archive.py` | on a built archive; extracts it and runs a launcher from the extraction |
| `check_host_kv.py` | from `check_profile_consistency.py` |
| `check_request_logs.py` | when a request log needs checking against its documented fields |
| `bump_test_baseline.py` | when a test is added and `suite_size` has to follow |
| `make_launchers_v3.py`, `generate_launcher_env.py` | regenerate launchers; never hand-edit a launcher |
| `verify_launchers_v3.py` | verify the generated launchers |
| `package_release.py` | cut and stage a release archive |
| `verify_shipping_artifacts.py` | re-verify published artifacts against their baselines |
| `probe_reasoning_effort.py` | map which `reasoning_effort` values the engine accepts |
| `repro_251.py`, `soak.py`, `compare_artifacts.py`, `per_domain_kl.py`, `test_per_domain_kl.py` | the prefix-state, soak and artifact-comparison instruments, each documented where its findings live |

`profiles.py` is the single source for what ships; `project_map.py` generates
[`docs/maintainer/project-map.md`](../docs/maintainer/project-map.md).

## Task index

| Task | Location |
|---|---|
| Convert weights with an official or custom recipe | [`convert/`](convert/); [user guide](../docs/weight-conversion.md) |
| Inspect artifact metadata and objects | [`artifact/inspect.py`](artifact/inspect.py) |
| One-time upgrade of official v2 artifacts | [`upgrade_ninfer_v2_to_v3.py`](upgrade_ninfer_v2_to_v3.py), with positional `INPUT OUTPUT` paths |
| Run benchmark matrices | [`bench/`](bench/README.md) |
| Measure external Serve TTFT | [`bench/ttft/`](bench/ttft/README.md) |
| Exercise a resident HTTP server | [`smoke/serve_contract.py`](smoke/serve_contract.py) |
| Exercise thinking preservation through a managed server | [`smoke/serve_thinking_preservation.py`](smoke/serve_thinking_preservation.py) |
| Measure sustained device-memory bandwidth | [`hbm_bandwidth_probe.cu`](hbm_bandwidth_probe.cu); [build command](#memory-bandwidth-probe) |
| Measure a serving lane's phase split and cache behaviour | [`bench/report_serve_phases.py`](bench/report_serve_phases.py) |
| Measure `prepared` and TTFT against conversation size | [`bench/warm_lane_sweep.py`](bench/warm_lane_sweep.py) |
| Decide whether a prefix cache serves a shared prefix, or only exact-request replay | [`bench/check_shared_prefix_reuse.py`](bench/check_shared_prefix_reuse.py) |
| Decide whether a first request is slower than later identical ones | [`bench/first_request_lane.py`](bench/first_request_lane.py) |

## Memory bandwidth probe

This probe is a CMake benchmark target, using the selected native CUDA architecture and benchmark
reference table. With a configured development tree (substitute `dgx-spark-dev` on Spark):

```bash
cmake --build build/rtx5090-dev -j --target ninfer_memory_bandwidth_bench
./build/rtx5090-dev/bench/ninfer_memory_bandwidth_bench
```

## Artifact workflow

The [conversion guide](../docs/weight-conversion.md) owns commands for official/custom recipes,
mixed sources, optional components, resources and sharding. The converter writes a `.ninfer`
artifact plus its `.conversion.json` report. Inspect a result with:

```bash
python3 -m tools.artifact.inspect out/qwen3_6_27b.ninfer --objects
```

Numeric formats, layouts and framing are defined by the references in the
[documentation map](../docs/README.md).

## Benchmark orchestration

Serving corpus benchmarks and concurrent saturation harnesses live under `tools/bench/`:

```bash
python3 tools/bench/run_serve_corpus.py --help
python3 tools/bench/run_serve_concurrency.py --help
```

See [`tools/bench/README.md`](bench/README.md) and [`bench/README.md`](../bench/README.md) for the
orchestrator and executable contracts.

For request-arrival latency, use the managed Qwen3.8-27B NVFP4/FP8 TTFT campaign. Its measurement
runner remains an external-only HTTP client; the separate controller owns Serve lifecycle and
artifacts. See [`tools/bench/ttft/README.md`](bench/ttft/README.md).

## Serving smoke

After starting `ninfer-serve` in another terminal:

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 \
  --model qwen3.6-27b
```

The client exercises OpenAI, Anthropic, streaming, usage, multimodal, and tool-call response
surfaces against the resident process.

For typed rewrite-checkpoint and thinking-history behavior, the managed smoke script launches a
real server and consumes the repository fixture:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --server-bin build/rtx5090-dev/apps/ninfer-serve \
  --artifact out/qwen3_6_27b.ninfer --backend mtp
```
