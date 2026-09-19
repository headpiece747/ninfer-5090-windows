# B200 (sm_100) Metric Name Reference

The stock `ncu_profile_skill.md` in older docs references metric names that **don't exist on B200 / sm_100**. This doc lists the actual names available in Nsight Compute 2026.1 on B200 and flags which names are different from older GPUs.

If a metric returns `None` on your kernel, first check this doc, then enumerate available names:

```python
action.metric_names()
```

---

## Metric names that changed

| Stock skill name (older GPU) | B200 / sm_100 name |
|---|---|
| `smsp__inst_executed_op_global_ld.sum` | **`smsp__sass_inst_executed_op_global_ld.sum`** |
| `smsp__inst_executed_op_global_st.sum` | **`smsp__sass_inst_executed_op_global_st.sum`** |
| `smsp__inst_executed_op_local_ld.sum` | **`smsp__sass_inst_executed_op_local_ld.sum`** |
| `smsp__inst_executed_op_local_st.sum` | **`smsp__sass_inst_executed_op_local_st.sum`** |
| `smsp__inst_executed_op_shared_ld.sum` | **`smsp__sass_inst_executed_op_shared_ld.sum`** |
| `smsp__inst_executed_op_shared_st.sum` | **`smsp__sass_inst_executed_op_shared_st.sum`** |
| `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` | (not available directly; compute from `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`) |
| `dram__bytes.sum` | (not directly; use `dram__bytes_read.sum + dram__bytes_write.sum`) |
| `sm__inst_executed_pipe_fmaheavy.*` | *not present on B200* — use `sm__inst_executed_pipe_fma.*` instead |
| `smsp__warps_issue_stalled_<reason>_per_issue_active.pct` | **`smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`** (note `average_` prefix and `ratio` suffix) |

---

## Canonical sm_100 metric set (curated)

These metric names have been confirmed to exist and return meaningful values on B200 / sm_100 with Nsight Compute 2026.1. Always verify for your specific ncu version by enumerating with `action.metric_names()` — NVIDIA occasionally renames metrics between releases.

### Launch geometry / occupancy
```
launch__grid_size
launch__block_size
launch__grid_dim_x, launch__grid_dim_y, launch__grid_dim_z
launch__block_dim_x, launch__block_dim_y, launch__block_dim_z
launch__thread_count
launch__waves_per_multiprocessor
launch__registers_per_thread
launch__shared_mem_per_block
launch__shared_mem_per_block_static
launch__shared_mem_per_block_dynamic
launch__occupancy_limit_blocks
launch__occupancy_limit_registers
launch__occupancy_limit_shared_mem
launch__occupancy_limit_warps
device__attribute_multiprocessor_count
device__attribute_max_warps_per_multiprocessor
sm__maximum_warps_per_active_cycle_pct              # theoretical occupancy %
```

### SOL (Speed-of-Light) / throughput
```
sm__throughput.avg.pct_of_peak_sustained_elapsed
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed
gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed
l1tex__throughput.avg.pct_of_peak_sustained_active
lts__throughput.avg.pct_of_peak_sustained_elapsed
dram__bytes_read.sum
dram__bytes_read.sum.pct_of_peak_sustained_elapsed
dram__bytes_read.sum.per_second                       # achieved BW
dram__bytes_write.sum
dram__bytes_write.sum.pct_of_peak_sustained_elapsed
dram__sectors_read.sum
dram__sectors_write.sum
```

### Timing
```
gpu__time_duration.sum                                # units: ns (check .unit())
smsp__cycles_active.avg
smsp__issue_active.avg.per_cycle_active               # issue rate per scheduler
smsp__issue_active.avg.pct_of_peak_sustained_active
```

### Warp activity
```
sm__warps_active.avg.pct_of_peak_sustained_active     # achieved occupancy %
sm__warps_active.avg.per_cycle_active
sm__warps_active.max.per_cycle_active
sm__warps_active.min.per_cycle_active
smsp__warps_active.avg.per_cycle_active               # per sub-partition
smsp__warps_eligible.avg.per_cycle_active
smsp__warps_eligible.max.per_cycle_active
```

### Compute pipelines
```
sm__inst_executed.avg.per_cycle_active                # IPC
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed
sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_elapsed
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed
sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_active

sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed
sm__pipe_tensor_subpipe_imma_cycles_active.avg.pct_of_peak_sustained_elapsed
sm__pipe_tensor_subpipe_dmma_cycles_active.avg.pct_of_peak_sustained_elapsed
sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.avg   # raw BF16→FP32 tensor ops
```

### Cache hit rates
```
l1tex__t_sector_hit_rate.pct                                       # overall L1
lts__t_sector_hit_rate.pct                                         # overall L2
l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct             # L1 hit on global loads
l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct             # L1 hit on global stores
```

### Memory access counts & sectors
```
smsp__sass_inst_executed_op_global_ld.sum                          # global LD instruction count
smsp__sass_inst_executed_op_global_st.sum                          # global ST count
smsp__sass_inst_executed_op_local_ld.sum                           # local LD (register spill)
smsp__sass_inst_executed_op_local_st.sum                           # local ST (register spill)
smsp__sass_inst_executed_op_shared.sum                             # total shared mem ops
smsp__sass_inst_executed_op_shared_ld.sum
smsp__sass_inst_executed_op_shared_st.sum

l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                     # L1 sectors for global LD
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                     # L1 sectors for global ST
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                    # LD request count
l1tex__t_requests_pipe_lsu_mem_global_op_st.sum                    # ST request count
# sectors/request = sectors.sum / requests.sum (ideal = 4 for 128B coalesced)
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio    # store efficiency (max 32)
```

### Stall reasons — aggregate ratios
```
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
smsp__average_warps_issue_stalled_membar_per_issue_active.ratio
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio
smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio
smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio
smsp__average_warps_issue_stalled_drain_per_issue_active.ratio
smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio
smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio
smsp__average_warps_issue_stalled_misc_per_issue_active.ratio
smsp__average_warps_issue_stalled_selected_per_issue_active.ratio       # productive (= 1.0)
```

### Stall reasons — per-PC (requires `--set source --section SourceCounters`)
```
smsp__pcsamp_sample_count                                           # total sample count
smsp__pcsamp_warps_issue_stalled_long_scoreboard                    # per-PC counts
smsp__pcsamp_warps_issue_stalled_short_scoreboard
smsp__pcsamp_warps_issue_stalled_wait
smsp__pcsamp_warps_issue_stalled_barrier
smsp__pcsamp_warps_issue_stalled_math_pipe_throttle
smsp__pcsamp_warps_issue_stalled_mio_throttle
smsp__pcsamp_warps_issue_stalled_lg_throttle
smsp__pcsamp_warps_issue_stalled_tex_throttle
smsp__pcsamp_warps_issue_stalled_not_selected
smsp__pcsamp_warps_issue_stalled_dispatch_stall
smsp__pcsamp_warps_issue_stalled_drain
smsp__pcsamp_warps_issue_stalled_no_instructions
smsp__pcsamp_warps_issue_stalled_selected
smsp__pcsamp_warps_issue_stalled_branch_resolving
smsp__pcsamp_warps_issue_stalled_membar
```

Each of these has `num_instances() > 0` with `correlation_ids()` that map to PCs. Use `action.source_info(pc)` to map PCs to `(file, line)`.

### PM sampling (time series)
```
pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg
pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg
pmsampling:smsp__warps_issue_stalled_wait.avg
pmsampling:smsp__warps_issue_stalled_dispatch_stall.avg
pmsampling:smsp__warps_issue_stalled_branch_resolving.avg
pmsampling:smsp__warps_issue_stalled_math_pipe_throttle.avg
pmsampling:smsp__warps_issue_stalled_mio_throttle.avg
pmsampling:smsp__warps_issue_stalled_lg_throttle.avg
pmsampling:smsp__warps_issue_stalled_no_instruction.avg
pmsampling:smsp__warps_issue_stalled_drain.avg
pmsampling:smsp__warps_issue_stalled_sleeping.avg
pmsampling:smsp__warps_issue_stalled_misc.avg
pmsampling:smsp__warps_issue_stalled_barrier.avg
pmsampling:smsp__warps_issue_stalled_tex_throttle.avg
```

Note: some `pmsampling:` metrics (notably `pmsampling:sm__throughput.*` and `pmsampling:dram__throughput.*`) may return empty instance arrays depending on ncu version / driver / GPU pair — always check `m.num_instances() > 0` before using. When SM/DRAM timelines are empty, the stall-reason timelines (`pmsampling:smsp__warps_issue_stalled_*`) usually tell the same story and are more reliably populated.

---

## Discovering metrics for new GPUs

When running on a GPU other than B200:

```bash
# All available metrics for a given chip
ncu --query-metrics --chip gb202        # B200 = GB202 / gb202 in some docs

# Filter by name pattern
ncu --query-metrics --chip gb202 | grep -i pmsampling
ncu --query-metrics --chip gb202 | grep -i issue_stalled

# Valid chip names: gb202 (B200), gh200 (H200 / Hopper), ga102 (Ampere), etc.
```

From Python, you can also enumerate per-report:
```python
all_names = action.metric_names()      # all metrics collected in this report
```

---

## Gotchas

1. **Metric exists in ncu's list but returns `None` from Python**: the metric wasn't *collected* in this report. Rerun ncu with the right `--section` or `--set`.
2. **Metric value is `0.0`**: either the hardware counter reports zero (e.g., no tensor core activity), or the metric is synthetic and depends on other metrics that weren't collected.
3. **`.avg` vs `.sum` vs `.max`**: each aggregate is a separate metric name. `.avg` is most commonly useful for rates / percentages; `.sum` for counts; `.max` for worst-case analysis.
4. **`pct_of_peak_sustained_elapsed` vs `pct_of_peak_sustained_active`**: `_elapsed` normalizes against total kernel time (including idle SMs); `_active` normalizes against cycles where the SM was actually running. `_elapsed` is more honest for under-utilized kernels.
