# `ncu_report` Python API

Use the Python module, not CLI output, for anything beyond a quick look. The API lets you extract, aggregate, compare, and archive metric data cleanly.

Module path (adjust for your CUDA version):
```bash
export PYTHONPATH=$PYTHONPATH:/usr/local/cuda-13.2/nsight-compute-2026.1.0/extras/python
python3 -c "import ncu_report; print('OK')"
```

---

## Basic loading

```python
import sys
sys.path.insert(0, "/usr/local/cuda-13.2/nsight-compute-2026.1.0/extras/python")
import ncu_report

report = ncu_report.load_report("path/to/full_<tag>.ncu-rep")

# A report can contain multiple "ranges" (each range = one profiled region).
# In practice, with -c 1 you have exactly one range containing one action (= one kernel launch).
rng = report.range_by_idx(0)
action = rng.action_by_idx(0)

print(f"Kernel demangled name: {action.name()}")       # void my_kernel<8, 256>(...)
print(f"Total metrics collected: {len(action.metric_names())}")
```

---

## Reading a single metric

```python
def safe(action, name, default=None):
    """Return metric value, or default if missing / errors."""
    try:
        return action[name].value()
    except Exception:
        return default

sm_util = safe(action, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
dram_read_bw = safe(action, "dram__bytes_read.sum.per_second")
duration_ns = safe(action, "gpu__time_duration.sum")     # usually in ns

print(f"SM throughput: {sm_util}%")
print(f"DRAM read BW:  {dram_read_bw/1e9:.2f} GB/s")
print(f"Duration:      {duration_ns/1e3:.2f} µs")
```

**Always wrap in a try/except or helper.** Metric names differ between GPU generations — see [`08-b200-metric-names.md`](08-b200-metric-names.md). A metric that exists on A100 may return `KeyError` on B200.

---

## Enumerating available metrics

```python
# Full list — 2000+ metrics for --set full
all_names = action.metric_names()

# Filter by pattern
for name in sorted(all_names):
    if "warps_issue_stalled" in name and "ratio" in name:
        print(name, "=", safe(action, name))
```

This is how you discover the *actual* metric names available on your GPU instead of guessing.

---

## Per-instance (per-SM, per-PC, per-time-sample) values

Many metrics have multiple values per collection. `value()` returns the aggregate (sum / avg depending on `rollup_operation`), but you can also enumerate the individual samples.

```python
m = action["pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg"]
n = m.num_instances()              # e.g., 1660 for a PM-sampled metric
print(f"instances: {n}")

vals = []
for i in range(n):
    try:
        v = m.as_double(i)
    except Exception:
        try:
            v = float(m.as_uint64(i))
        except Exception:
            v = None
    vals.append(v)
```

For PM sampling this is the timeline — index `i` is a time-ordered sample. Bucket these for an ASCII plot (see `helpers/plot_timeline.py`).

---

## Per-PC → per-source-line mapping

The source-level report (collected with `--set source --section SourceCounters`) has per-PC samples. Map them to source lines via `action.source_info(pc)`:

```python
def per_pc_stalls(action, stall_metric):
    m = action[stall_metric]
    n = m.num_instances()
    if n == 0 or not m.has_correlation_ids():
        return []
    cor = m.correlation_ids()
    out = []
    for i in range(n):
        pc = cor.as_uint64(i)
        val = m.as_uint64(i)
        si = action.source_info(pc)
        if si is None:
            file, line = "?", 0
        else:
            file, line = si.file_name(), si.line()
        out.append((file, line, val))
    return out

stalls = per_pc_stalls(action, "smsp__pcsamp_warps_issue_stalled_long_scoreboard")
```

Aggregate by `(file, line)` and sort by total to get hottest stall lines. See `helpers/extract_stall_hotspots.py` for a complete implementation.

---

## Discovering Value Kind

Each metric has a value kind (uint64, double, float, string). Use `m.kind()` to check before calling the right accessor:

```python
def metric_val(m, i=None):
    k = m.kind()
    VK = m.ValueKind_UINT64, m.ValueKind_DOUBLE, m.ValueKind_FLOAT, m.ValueKind_STRING
    if i is None:
        return m.value()          # aggregate
    if k == m.ValueKind_UINT64:
        return m.as_uint64(i)
    if k in (m.ValueKind_DOUBLE, m.ValueKind_FLOAT):
        return m.as_double(i)
    if k == m.ValueKind_STRING:
        return m.as_string(i)
    # Try generic conversions as fallbacks
    try:
        return m.as_uint64(i)
    except Exception:
        return m.as_double(i)
```

---

## Useful `action` / `metric` methods

```python
# Action (= one kernel launch's profile data)
action.name()                   # demangled kernel name
action.metric_names()           # list of all metrics
action.metric_by_name(name)     # same as action[name]
action.source_info(pc)          # IPC → SourceInfo (file, line) — only if -lineinfo was used
action.sass_by_pc()             # dict {pc → SASS instruction string}
action.ptx_by_pc()              # dict {pc → PTX} — only if --keep was used
action.rule_results_as_dicts()  # NCU rule-engine output as list of dicts

# Metric
m.value()                       # aggregate value
m.unit()                        # string, e.g. "%" or "cycle"
m.kind()                        # value kind (UINT64 / DOUBLE / ...)
m.rollup_operation()            # AVG / MAX / MIN / SUM / NONE
m.num_instances()               # per-instance count (0 if aggregate only)
m.has_correlation_ids()         # True for per-PC metrics
m.correlation_ids()             # parallel array for num_instances
m.description()                 # human-readable metric description
```

---

## Exploring when you don't know the right metric name

```python
# Print all metric names sorted
for n in sorted(action.metric_names()):
    print(n)

# Print metrics matching a pattern with their current value
import re
pat = re.compile(r"dram__bytes.*sum")
for n in sorted(action.metric_names()):
    if pat.search(n):
        try:
            v = action[n].value()
            print(f"{n} = {v}  (unit: {action[n].unit()})")
        except Exception as e:
            print(f"{n} = ERROR {e}")
```

This is how I built [`08-b200-metric-names.md`](08-b200-metric-names.md) — by enumerating everything available on sm_100.

---

## Comparing two reports programmatically

```python
def compare(rep1_path, rep2_path, metrics):
    r1 = ncu_report.load_report(rep1_path)
    r2 = ncu_report.load_report(rep2_path)
    a1 = r1.range_by_idx(0).action_by_idx(0)
    a2 = r2.range_by_idx(0).action_by_idx(0)
    print(f"{'Metric':<75} {'v1':>15} {'v2':>15} {'change':>10}")
    for m in metrics:
        v1 = safe(a1, m)
        v2 = safe(a2, m)
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) and v1:
            chg = (v2 - v1) / v1 * 100
            print(f"{m:<75} {v1:>15.4g} {v2:>15.4g} {chg:>+9.1f}%")
        else:
            print(f"{m:<75} {str(v1):>15} {str(v2):>15}")

compare("v1.ncu-rep", "v2.ncu-rep", [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "l1tex__t_sector_hit_rate.pct",
])
```

---

## Extracting NCU's rule suggestions

The rule engine results — the "OPT Est. Speedup: X%" bullets — are accessible as structured data:

```python
for result in action.rule_results_as_dicts():
    severity = result.get("type", "?")      # "OPT" / "INF" / "WRN"
    rule_name = result.get("rule_name", "?")
    message = result.get("message_for_display", "?")
    est_speedup = result.get("estimated_speedup_pct", None)
    print(f"[{severity}] {rule_name}: {est_speedup}%")
    print(f"    {message[:200]}")
```

Sort by `estimated_speedup_pct` descending to get the highest-impact suggestions first.

---

## Saving everything for later

Always archive the full metric dump:

```python
import json
from pathlib import Path

def dump_all(action, outpath):
    rows = []
    for name in sorted(action.metric_names()):
        try:
            m = action[name]
            rows.append({
                "name": name,
                "value": m.value(),
                "unit": m.unit() if hasattr(m, "unit") else "",
            })
        except Exception as e:
            rows.append({"name": name, "error": str(e)})
    Path(outpath).write_text(json.dumps(rows, indent=1, default=str))

dump_all(action, "analysis/metrics_all_<tag>.json")
```

This makes future re-analysis cheap: the raw data lives as JSON, you don't need to reopen the `.ncu-rep`.

---

## Gotchas

- **`KeyError` on a metric that "should" exist**: the metric has a different name on this GPU. Check [`08-b200-metric-names.md`](08-b200-metric-names.md) or enumerate with `action.metric_names()`.
- **`num_instances() == 0`** but you expected per-instance data: the metric wasn't collected in instanced mode, or the section that produces it wasn't requested. Re-run ncu with the right `--section`.
- **`has_correlation_ids() == False`** on a source-level metric: `-lineinfo` wasn't on the compile line. Rebuild.
- **`source_info(pc)` returns None**: same as above — rebuild with `-lineinfo`.
- **Metric value is a string** like `"PolicySpread"`: it's an enum, use `m.as_string()` or `m.value()` and expect a string.
