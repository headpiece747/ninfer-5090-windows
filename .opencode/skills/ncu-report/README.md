# ncu-report-skill
> [!IMPORTANT]
> This skill is maintained as a standalone submodule of
> [Kernel Design Agents (KDA)](https://github.com/mit-han-lab/kernel-design-agents)
> for easy installation.
>
> For bug reports, feature requests, and discussions, please use the main KDA repository:
> https://github.com/mit-han-lab/kernel-design-agents

A Claude Code skill for profiling CUDA kernels with Nsight Compute on NVIDIA B200 (sm_100). Covers the full workflow: build a standalone harness, run `ncu`, parse reports with the Python API, walk through six analysis dimensions, match patterns to a diagnosis playbook, and write an evidence-backed optimization report.

The skill is self-contained: reference docs, reusable helper scripts (harness template, safetensors loader, report-analysis Python), and a companion Blackwell programming reference all ship in this repo.

---

## What's in this repo

```
.
├── SKILL.md                          ← skill entry point (with YAML frontmatter)
├── helpers/                          ← reusable code
│   ├── harness_template.cu           ← standalone profiling harness template
│   ├── safetensors_loader.h          ← header-only safetensors reader (no deps)
│   ├── list_flashinfer_workloads.py  ← browse flashinfer-trace datasets
│   ├── analyze_reports.py            ← extract + compare key metrics from .ncu-rep files
│   ├── extract_stall_hotspots.py     ← per-line stall aggregation (source-level reports)
│   ├── plot_timeline.py              ← ASCII PM-sampling timeline plots (reveals tail effects)
│   ├── ncu_utils.py                  ← shared Python helpers, B200-compatible key metric list
│   └── README.md
├── reference/                        ← detailed reference docs
│   ├── 00-directory-layout.md        ← profile/ directory conventions (read first)
│   ├── 01-workflow.md                ← end-to-end profiling checklist
│   ├── 02-harness-guide.md           ← how to build a standalone profiling harness
│   ├── 03-collection.md              ← ncu command recipes
│   ├── 04-python-api.md              ← ncu_report Python API patterns
│   ├── 05-analysis-dimensions.md     ← six analysis dimensions
│   ├── 06-diagnosis-playbook.md      ← pattern → cause → fix
│   ├── 07-report-template.md         ← final report structure
│   ├── 08-b200-metric-names.md       ← sm_100 metric name reference
│   └── 09-common-issues.md           ← permissions, PM sampling, JIT toolchains, etc.
└── blackwell-cuda-programming.md     ← companion reference: Blackwell programming principles
```

---

## Installation

Claude Code discovers skills in two locations:

- `~/.claude/skills/<skill_name>/SKILL.md` — **user-level**, available in every project
- `<repo>/.claude/skills/<skill_name>/SKILL.md` — **project-level**, scoped to one repo

Three ways to install this skill:

### Option 1 — Symlink from a clone (recommended)

Keeps the skill version-controlled and easy to update; edits in the clone are picked up instantly.

```bash
# Clone somewhere stable
git clone git@github.com:DongyunZou/ncu-report-skill.git ~/workspace/ncu-report-skill

# User-level install: make the skill available in every project
mkdir -p ~/.claude/skills
ln -s ~/workspace/ncu-report-skill ~/.claude/skills/ncu-report-skill

# Or project-level install: scope to one repo
cd /path/to/other-repo
mkdir -p .claude/skills
ln -s ~/workspace/ncu-report-skill .claude/skills/ncu-report-skill
```

Pull updates with `cd ~/workspace/ncu-report-skill && git pull`. The symlinks pick up the new content automatically.

### Option 2 — Copy into place

If you prefer a static copy over a symlink:

```bash
git clone git@github.com:DongyunZou/ncu-report-skill.git /tmp/ncu
mkdir -p ~/.claude/skills
cp -r /tmp/ncu ~/.claude/skills/ncu-report-skill
```

### Option 3 — Git submodule (for a project-level install committed alongside the repo)

```bash
cd /path/to/other-repo
git submodule add git@github.com:DongyunZou/ncu-report-skill.git .claude/skills/ncu-report-skill
git commit -m "Add ncu-report-skill as a submodule"
```

---

## How Claude uses this skill

Once installed at `~/.claude/skills/ncu-report-skill/` (or project-level), Claude Code will:

1. Advertise the skill's name + description in the system reminder of new conversations.
2. Let the user invoke it manually via `/ncu-report-skill` or let the model invoke it with the Skill tool when the conversation matches the `description` triggers.

When invoked, Claude reads `SKILL.md`, follows its workflow (phases 0 → 6 in `reference/01-workflow.md`), and uses the helper scripts in `helpers/` as needed.

---

## Running the helpers directly (no Claude needed)

The Python helpers work standalone for any `.ncu-rep` you have:

```bash
# Make sure ncu_report is importable (the helpers try common paths automatically)
export PYTHONPATH=$PYTHONPATH:/usr/local/cuda-13.2/nsight-compute-2026.1.0/extras/python

# Create a run directory
export RUN=/path/to/your/profile/myrun

# Extract key metrics from one or more reports
python3 ~/.claude/skills/ncu-report-skill/helpers/analyze_reports.py \
    --run-dir "$RUN" \
    --report "$RUN/reports/full_<tag>.ncu-rep" --tag <tag>

# Per-line stall hotspots (requires a source-level .ncu-rep)
python3 ~/.claude/skills/ncu-report-skill/helpers/extract_stall_hotspots.py \
    --run-dir "$RUN" \
    --report "$RUN/reports/source_<tag>.ncu-rep" --tag <tag>

# ASCII PM-sampling timelines
python3 ~/.claude/skills/ncu-report-skill/helpers/plot_timeline.py \
    --run-dir "$RUN" \
    --report "$RUN/reports/full_<tag>.ncu-rep" --tag <tag>

# Browse a flashinfer-trace dataset to pick workload shapes
export FIB_DATASET_PATH=/path/to/flashinfer-trace
python3 ~/.claude/skills/ncu-report-skill/helpers/list_flashinfer_workloads.py \
    --definition <your_definition_name>
```

The C++ harness template + safetensors loader live under `helpers/`; copy them into your profile run's `harness/` directory and fill in the kernel body. See `reference/02-harness-guide.md` for details.

---

## Requirements

- CUDA Toolkit with `nvcc` (tested with 13.2)
- Nsight Compute CLI `ncu` (tested with 2026.1)
- The `ncu_report` Python module (ships with Nsight Compute under `extras/python/`)
- An NVIDIA GPU with permission to access performance counters (see `reference/09-common-issues.md` if `ncu` reports `ERR_NVGPUCTRPERM`)

The skill is optimized for B200 / sm_100 metric names, but the workflow and helpers work on any CUDA GPU Nsight Compute supports. Metric names may differ on older GPUs (A100, H100) — see `reference/08-b200-metric-names.md` for guidance.

---

## License

MIT
