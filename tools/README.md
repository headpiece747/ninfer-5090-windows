# NInfer tools

`tools/` contains artifact conversion and inspection, benchmark orchestration, and serving smoke
checks. To download and run an existing artifact, start with the [project README](../README.md).
To build your own weights, use the [weight conversion guide](../docs/weight-conversion.md).

Run commands from the repository root with a Python environment containing the dependencies
for the selected tool. The maintained environment uses Python 3.11.

Python tools are independent of CMake; there is no `NINFER_BUILD_TOOLS` option.

## Release tools

`release/` holds the gates and the packaging pipeline. The six that run on every commit are wired
into `.githooks/pre-commit`, and [the row for that hook in `AGENTS.md`](../AGENTS.md) is the
authority for what they cover:

| Tool | Runs |
|---|---|
| `check_doc_links.py` | every commit, and again during packaging |
| `check_text_encoding.py` | every commit |
| `check_fp8_band_ladders.py` | every commit; its `--self-test` is the control |
| `check_profile_consistency.py` | every commit, and again during packaging |
| `check_calibration_corpus.py` | every commit |
| `check_production_stream_defaults.py` | every commit |
| `check_rule_count.py` | every commit; keeps `AGENTS.md`'s stated rule count equal to the bullets |
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
| Measure the physical HBM read/copy ceiling | [`hbm_bandwidth_probe.cu`](hbm_bandwidth_probe.cu); [build command](#standalone-hbm-probe) |
| Measure a serving lane's phase split and cache behaviour | [`bench/report_serve_phases.py`](bench/report_serve_phases.py) |
| Measure `prepared` and TTFT against conversation size | [`bench/warm_lane_sweep.py`](bench/warm_lane_sweep.py) |
| Decide whether a first request is slower than later identical ones | [`bench/first_request_lane.py`](bench/first_request_lane.py) |

## Standalone HBM probe

This maintainer probe has an explicit standalone CUDA build, independent of the CMake benchmark
targets. Build it with the project's CUDA toolkit and run it from the repository root:

```bash
mkdir -p build
nvcc -O3 -std=c++17 -arch=sm_120a tools/hbm_bandwidth_probe.cu \
  -o build/hbm_bandwidth_probe
./build/hbm_bandwidth_probe
```

## Artifact workflow

The common converter reads selected local sources and writes a `.ninfer` artifact plus its
`.conversion.json` report. These examples include the optional weights used by the official
artifacts. The input paths are placeholders for local checkpoint checkouts:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.6-27B \
  --recipe qwen3_6_27b --components text,vision,mtp --proposal \
  --resource chat_template.jinja=tools/chat_templates/qwen3_6.jinja \
  --name qwen3.6-27b \
  --out out/qwen3_6_27b.ninfer

python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B \
  --recipe qwen3_8_27b --components text,vision,mtp,dflash2 --proposal \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --name qwen3.8-27b \
  --out out/qwen3_8_27b.ninfer

python3 -m tools.convert \
  --model /path/to/Qwen3.6-35B-A3B-base \
  --recipe qwen3_6_35b_a3b --components text,vision,mtp,dflash --proposal \
  --source dflash=/path/to/Qwen3.6-35B-A3B-DFlash \
  --resource chat_template.jinja=tools/chat_templates/qwen3_6.jinja \
  --name qwen3.6-35b-a3b \
  --out out/qwen3_6_35b_a3b.ninfer
```

Inspect a result:

```bash
python3 -m tools.artifact.inspect out/qwen3_6_27b.ninfer --objects
```

Recipes, mixed sources, custom methods, resources and sharding are described in the
[conversion guide](../docs/weight-conversion.md). Numeric formats, layouts and framing are defined
by the references linked from the [documentation map](../docs/README.md).

## Benchmark orchestration

`tools/bench/run_ninfer_bench_matrix.py` builds and runs the public-Engine benchmark matrix and
writes ignored local reports below `profiles/bench/`:

```bash
python3 tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run
python3 tools/bench/run_ninfer_bench_matrix.py --preset core
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
  --artifact out/qwen3_6_27b.ninfer --backend mtp
```
