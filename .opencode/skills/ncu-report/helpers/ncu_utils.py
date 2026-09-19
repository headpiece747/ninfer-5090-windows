"""Shared helpers for parsing Nsight Compute reports.

Usage:
    from ncu_utils import load_report, safe, dump_all_metrics

The caller is expected to have set PYTHONPATH to include ncu_report, e.g.:
    export PYTHONPATH=$PYTHONPATH:/usr/local/cuda-13.2/nsight-compute-2026.1.0/extras/python

If ncu_report is not importable, we try a small list of common paths.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# --- Attempt to locate ncu_report --------------------------------------------
def _locate_ncu_report():
    candidates = [
        "/usr/local/cuda-13.2/nsight-compute-2026.1.0/extras/python",
        "/usr/local/cuda/nsight-compute/extras/python",
    ]
    # Also probe /usr/local/cuda-*/nsight-compute-*/extras/python
    for root in ["/usr/local", "/opt/nvidia", "/opt/cuda"]:
        p = Path(root)
        if not p.is_dir():
            continue
        for sub in p.glob("cuda-*/nsight-compute-*/extras/python"):
            candidates.append(str(sub))
        for sub in p.glob("nsight-compute-*/extras/python"):
            candidates.append(str(sub))
    for c in candidates:
        if Path(c).is_dir() and (Path(c) / "ncu_report.py").exists():
            return c
    return None

try:
    import ncu_report  # noqa: F401
except ImportError:
    found = _locate_ncu_report()
    if found:
        sys.path.insert(0, found)
        import ncu_report  # noqa: F401
    else:
        raise


import ncu_report  # noqa: E402


# --- Loading -----------------------------------------------------------------

def load_report(path):
    """Load a .ncu-rep file and return the first action (= first kernel launch).

    Returns (report, action) tuple — keep the report alive while using the action.
    """
    r = ncu_report.load_report(str(path))
    rng = r.range_by_idx(0)
    action = rng.action_by_idx(0)
    return r, action


def load_action(path):
    """Shortcut when you don't need the report object separately."""
    _, action = load_report(path)
    return action


# --- Safe metric access ------------------------------------------------------

def safe(action, name, default=None):
    """Return metric value, or `default` if the metric is missing or errors."""
    try:
        return action[name].value()
    except Exception:
        return default


def safe_many(action, names, default=None):
    """Bulk-fetch multiple metrics. Returns a dict name -> value-or-default."""
    return {n: safe(action, n, default) for n in names}


def metric_or_none(action, *candidates):
    """Try each candidate name, return first that works. Useful for
    GPU-gen-specific names: some metric names differ on sm_100 vs sm_90."""
    for n in candidates:
        v = safe(action, n, None)
        if v is not None:
            return v
    return None


# --- Value-kind robust accessor (for per-instance data) ---------------------

def metric_value_at(m, i):
    """Read the i-th instance value regardless of value kind."""
    k = m.kind()
    if k == m.ValueKind_UINT64:
        return m.as_uint64(i)
    if k in (m.ValueKind_DOUBLE, m.ValueKind_FLOAT):
        return m.as_double(i)
    if k == m.ValueKind_STRING:
        return m.as_string(i)
    # Fallbacks
    try:
        return m.as_uint64(i)
    except Exception:
        try:
            return m.as_double(i)
        except Exception:
            return None


def per_instance_values(action, metric_name):
    """Return a list of per-instance values, or None if the metric has none."""
    try:
        m = action[metric_name]
    except Exception:
        return None
    try:
        n = m.num_instances()
    except Exception:
        return None
    if n == 0:
        return None
    return [metric_value_at(m, i) for i in range(n)]


# --- Archive all metrics -----------------------------------------------------

def dump_all_metrics(action, outfile):
    """Dump every metric name + value to a JSON file for later analysis.

    Returns the number of entries written.
    """
    out = []
    for n in sorted(action.metric_names()):
        try:
            m = action[n]
            rec = {"name": n}
            try:
                rec["value"] = m.value()
            except Exception as e:
                rec["error"] = str(e)
            try:
                rec["unit"] = m.unit()
            except Exception:
                pass
            out.append(rec)
        except Exception as e:
            out.append({"name": n, "error": str(e)})
    Path(outfile).write_text(json.dumps(out, indent=1, default=str))
    return len(out)


# --- PC → source line mapping ------------------------------------------------

def per_pc_values(action, metric_name):
    """For a source-level metric (with correlation_ids = PCs), return list of (pc, value)."""
    try:
        m = action[metric_name]
    except Exception:
        return []
    try:
        n = m.num_instances()
    except Exception:
        return []
    if n == 0 or not m.has_correlation_ids():
        return []
    cor = m.correlation_ids()
    out = []
    for i in range(n):
        try:
            pc = cor.as_uint64(i)
        except Exception:
            try:
                pc = int(cor.as_double(i))
            except Exception:
                pc = None
        try:
            v = metric_value_at(m, i)
        except Exception:
            v = 0
        out.append((pc, v))
    return out


def pc_to_source_line(action, pc):
    """Return (file, line) for a given PC, or ('?', 0) if unavailable.

    Requires -lineinfo at compile time.
    """
    try:
        si = action.source_info(pc)
        if si is None:
            return "?", 0
        return si.file_name(), si.line()
    except Exception:
        return "?", 0


# --- Curated metric sets -----------------------------------------------------
#
# These metric names are known to exist and return meaningful values on
# B200 / sm_100 with Nsight Compute 2026.x. For a fuller list and rationale
# see ../reference/08-b200-metric-names.md. Other GPU generations (A100, H100,
# consumer cards) and future ncu releases may need alternate names — always
# verify with action.metric_names() if a metric returns None.

B200_KEY_METRICS = [
    # Launch geometry
    "launch__grid_size",
    "launch__block_size",
    "launch__grid_dim_x",
    "launch__grid_dim_y",
    "launch__grid_dim_z",
    "launch__block_dim_x",
    "launch__waves_per_multiprocessor",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "launch__thread_count",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "device__attribute_multiprocessor_count",
    "device__attribute_max_warps_per_multiprocessor",
    # Timing
    "gpu__time_duration.sum",
    "smsp__cycles_active.avg",
    # SOL
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    # Occupancy
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.per_cycle_active",
    "sm__warps_active.max.per_cycle_active",
    "sm__warps_active.min.per_cycle_active",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warps_eligible.max.per_cycle_active",
    # IPC
    "sm__inst_executed.avg.per_cycle_active",
    "smsp__issue_active.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.avg",
    # Compute pipes
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_active",
    # Tensor core
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.avg",
    # DRAM
    "dram__bytes_read.sum",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum.per_second",
    "dram__bytes_write.sum",
    "dram__bytes_write.sum.pct_of_peak_sustained_elapsed",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
    # Caches
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
    "l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct",
    # Memory instruction counts
    "smsp__sass_inst_executed_op_global_ld.sum",
    "smsp__sass_inst_executed_op_global_st.sum",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
    "smsp__sass_inst_executed_op_shared.sum",
    "smsp__sass_inst_executed_op_shared_ld.sum",
    "smsp__sass_inst_executed_op_shared_st.sum",
    # Sectors / requests (for coalescing analysis)
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio",
    # Stall reasons — aggregate ratios
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_misc_per_issue_active.ratio",
    # Stall reasons — per-PC (requires --set source)
    "smsp__pcsamp_sample_count",
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_wait",
    "smsp__pcsamp_warps_issue_stalled_barrier",
    "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle",
    "smsp__pcsamp_warps_issue_stalled_mio_throttle",
    "smsp__pcsamp_warps_issue_stalled_lg_throttle",
    "smsp__pcsamp_warps_issue_stalled_not_selected",
    "smsp__pcsamp_warps_issue_stalled_dispatch_stall",
    "smsp__pcsamp_warps_issue_stalled_drain",
    "smsp__pcsamp_warps_issue_stalled_no_instructions",
    "smsp__pcsamp_warps_issue_stalled_selected",
    "smsp__pcsamp_warps_issue_stalled_branch_resolving",
    "smsp__pcsamp_warps_issue_stalled_membar",
]


# --- Convenience: NCU rule results --------------------------------------------

def rule_results(action):
    """Return the NCU rule-engine results as a list of dicts, or [] if unavailable."""
    try:
        return list(action.rule_results_as_dicts())
    except Exception:
        return []


def rule_speedups(action):
    """Return list of (est_speedup_pct, rule_name, message) sorted desc by est_speedup.
    Missing est_speedup becomes 0."""
    out = []
    for rr in rule_results(action):
        est = rr.get("estimated_speedup_pct", None)
        if est is None:
            est = 0.0
        try:
            est = float(est)
        except Exception:
            est = 0.0
        rule = rr.get("rule_name", "?")
        msg = rr.get("message_for_display", "?")
        out.append((est, rule, msg))
    out.sort(key=lambda x: -x[0])
    return out
