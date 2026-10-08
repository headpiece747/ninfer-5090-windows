"""Executable Serve profiles for the external TTFT campaigns."""

from __future__ import annotations

from decimal import Decimal


# qwen3_8_27b_nvfp4: StateImageHostLayout sums 48 GDN conv/state regions and
# continuation hidden, each aligned to 256 B (state/state_image.cpp). DFlash2 also
# carries 5 local K/V layers of 2048 positions. MTP Full KV is in the KV arena.
STATE_IMAGE_BYTES = 153_954_304
DFLASH2_STATE_IMAGE_BYTES = 195_897_344


def _host_mib(capacity_bytes: int) -> str:
    return str(Decimal(capacity_bytes) / Decimal(1 << 20))


def _args(*values: str | int) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


COMMON_ARGS = _args(
    "--prefill-chunk",
    "8192",
    "--kv-dtype",
    "fp8",
    "--no-thinking",
    "--greedy",
    "--log-stats-interval-ms",
    1000,
)


PROFILE_ARGS: dict[str, tuple[str, ...]] = {
    "text-cold-8k": _args(
        "--max-context", 8192,
        "--kv-capacity", 8192,
        "--max-concurrency", 1,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "text-cold-64k": _args(
        "--max-context", 65536,
        "--kv-capacity", 65536,
        "--max-concurrency", 1,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "text-cold-256k": _args(
        "--max-context", 262144,
        "--kv-capacity", 262144,
        "--max-concurrency", 1,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "cache-hot": _args(
        "--max-context", 8192,
        "--kv-capacity", 8192,
        "--max-concurrency", 1,
        "--device-state-slots", 2,
        "--host-context-mib", 0,
    ),
    "cache-state-working-set": _args(
        "--max-context", 32768,
        "--kv-capacity", 32768,
        "--max-concurrency", 1,
        "--prefill-chunk", 1024,
        "--spec", "dflash2",
        "--draft-tokens", 7,
        "--lm-head-draft",
        "--device-state-slots", 8,
        "--host-context-mib", _host_mib(8 * DFLASH2_STATE_IMAGE_BYTES),
    ),
    "cache-private-working-set": _args(
        "--max-context", 32768,
        "--kv-capacity", 32768,
        "--max-concurrency", 1,
        "--prefill-chunk", 1024,
        "--spec", "dflash2",
        "--draft-tokens", 7,
        "--lm-head-draft",
        "--device-state-slots", 2,
        "--host-context-mib", _host_mib(2 * DFLASH2_STATE_IMAGE_BYTES),
    ),
    "cache-pressure-device": _args(
        "--max-context", 8192,
        "--kv-capacity", 16384,
        "--max-concurrency", 2,
        "--device-state-slots", 2,
        "--host-context-mib", 0,
    ),
    "cache-pressure-state-host": _args(
        "--max-context", 8192,
        "--kv-capacity", 16384,
        "--max-concurrency", 2,
        "--device-state-slots", 0,
        "--host-context-mib", _host_mib(4 * STATE_IMAGE_BYTES),
    ),
    "cache-pressure-kv-host": _args(
        "--max-context", 8192,
        "--kv-capacity", 8192,
        "--max-concurrency", 2,
        "--device-state-slots", 2,
        "--host-context-mib", 8192,
    ),
    "cache-swap-64k-host": _args(
        "--max-context", 65536,
        "--kv-capacity", 65536,
        "--max-concurrency", 2,
        "--device-state-slots", 4,
        "--host-context-mib", 4608,
    ),
    "cache-rotation-55k-host": _args(
        "--max-context", 240000,
        "--kv-capacity", 240000,
        "--max-concurrency", 4,
        "--max-pending-requests", 32,
        "--pending-timeout-ms", 120000,
        "--spec", "mtp",
        "--draft-tokens", 3,
        "--lm-head-draft",
        "--device-state-slots", 2,
        "--host-context-mib", _host_mib((49152 << 20) + 24 * STATE_IMAGE_BYTES),
    ),
    "cache-pressure-both-host": _args(
        "--max-context", 8192,
        "--kv-capacity", 8192,
        "--max-concurrency", 2,
        "--device-state-slots", 0,
        "--host-context-mib", _host_mib((8192 << 20) + 4 * STATE_IMAGE_BYTES),
    ),
    "cache-pressure-evict": _args(
        "--max-context", 8192,
        "--kv-capacity", 8192,
        "--max-concurrency", 2,
        "--device-state-slots", 1,
        "--host-context-mib", 0,
    ),
    # The reporter's production configuration from the port's issue 5, scaled to fit this machine and
    # translated onto the post-b9114396 CLI. Upstream's context-cache replacement collapsed the
    # explicit capacity flags this profile was first written against -- `--host-state-slots`,
    # `--host-kv-mib`, `--max-shared-prefixes`, `--max-private-continuations` and
    # `--max-long-anchors-per-continuation` no longer exist -- so host state and host KV are now one
    # `--host-context-mib` quota and the descriptor counts are derived. What the case exercises is
    # preserved: three concurrent lanes, state parked to host and restored, and sessions far longer
    # than any default ceiling.
    #
    # Their literals do not fit this machine: the engine refused startup with
    # "requires 20316679168 bytes, but only 15289286656 bytes are available", because host KV alone is
    # 24 GiB beside 16 GiB of weights on a 48 GB box. Every quantity is therefore scaled by the same
    # ratio rather than tuned one at a time. 40% of their values: kv-capacity 294912 -> 117964,
    # host KV 24576 MiB -> 9830 MiB, max-context 262144 -> 104857, which still leaves every session
    # room to grow past 8,000 tokens. Their unscaled and verbatim lines need 20.3 GB of runtime and
    # are recorded in docs/research/resource-underflow-issue-5.md rather than shipped as profiles
    # that cannot start here.
    "cache-reporter-concurrency3-scaled": _args(
        "--max-context", 104857,
        "--kv-capacity", 117964,
        "--prefill-chunk", 1024,
        "--max-concurrency", 3,
        "--max-pending-requests", 32,
        "--pending-timeout-ms", 3600000,
        "--device-state-slots", 2,
        "--host-context-mib", _host_mib(16 * STATE_IMAGE_BYTES + 9830 * (1 << 20)),
        "--preserve-thinking",
    ),
    "cache-off": _args(
        "--max-context", 8192,
        "--kv-capacity", 8192,
        "--max-concurrency", 1,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "shared-prefix": _args(
        "--max-context", 8192,
        "--kv-capacity", 16384,
        "--max-concurrency", 2,
        "--device-state-slots", 2,
        "--host-context-mib", 0,
    ),
    "prefix-competition": _args(
        "--max-context", 16384,
        "--kv-capacity", 16384,
        "--max-concurrency", 1,
        "--device-state-slots", 3,
        "--host-context-mib", 0,
    ),
    "session-order": _args(
        "--max-context", 8192,
        "--kv-capacity", 16384,
        "--max-concurrency", 2,
        "--device-state-slots", 8,
        "--host-context-mib", 0,
    ),
    "scheduler-overlap": _args(
        "--max-context", 8192,
        "--kv-capacity", 16384,
        "--max-concurrency", 2,
        "--prefill-chunk", 1024,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "scheduler-prefill-128": _args(
        "--max-context", 8192,
        "--kv-capacity", 16384,
        "--max-concurrency", 2,
        "--prefill-chunk", 128,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "scheduler-prefill-4096": _args(
        "--max-context", 8192,
        "--kv-capacity", 16384,
        "--max-concurrency", 2,
        "--prefill-chunk", 4096,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "scheduler-kv-pressure": _args(
        "--max-context", 7744,
        "--kv-capacity", 7808,
        "--max-concurrency", 2,
        "--pending-timeout-ms", 120000,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "lane-limit-8": _args(
        "--max-context", 4224,
        "--kv-capacity", 33792,
        "--max-concurrency", 8,
        "--max-pending-requests", 1,
        "--pending-timeout-ms", 120000,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "pending-timeout": _args(
        "--max-context", 4224,
        "--kv-capacity", 4224,
        "--max-concurrency", 1,
        "--max-pending-requests", 1,
        "--pending-timeout-ms", 100,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "context-boundary": _args(
        "--max-context", 8192,
        "--kv-capacity", 8192,
        "--max-concurrency", 1,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "vision-cache": _args(
        "--max-context", 32768,
        "--kv-capacity", 32768,
        "--max-concurrency", 1,
        "--device-state-slots", 2,
        "--vision",
        "--media-cache-mib", 512,
        "--media-live-mib", 512,
        "--host-context-mib", 0,
    ),
    "vision-thread-1": _args(
        "--max-context", 32768,
        "--kv-capacity", 32768,
        "--max-concurrency", 1,
        "--device-state-slots", 2,
        "--vision",
        "--media-cache-mib", 512,
        "--media-live-mib", 512,
        "--media-preprocess-threads", 1,
        "--host-context-mib", 0,
    ),
    "vision-concurrent": _args(
        "--max-context", 32768,
        "--kv-capacity", 65536,
        "--max-concurrency", 2,
        "--vision",
        "--media-cache-mib", 512,
        "--media-live-mib", 1024,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "media-cache-tight": _args(
        "--max-context", 8192,
        "--kv-capacity", 8192,
        "--max-concurrency", 1,
        "--vision",
        "--media-cache-mib", 16,
        "--media-live-mib", 128,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "vision-boundary": _args(
        "--max-context", 65536,
        "--kv-capacity", 65536,
        "--max-concurrency", 1,
        "--vision",
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "mixed-four": _args(
        "--max-context", 8192,
        "--kv-capacity", 32768,
        "--max-concurrency", 4,
        "--vision",
        "--host-context-mib", _host_mib((8192 << 20) + 8 * STATE_IMAGE_BYTES),
    ),
    "preemption-replay": _args(
        "--max-context", 512,
        "--kv-capacity", 512,
        "--max-concurrency", 2,
        "--prefill-chunk", 128,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "preemption-snapshot": _args(
        "--max-context", 512,
        "--kv-capacity", 512,
        "--max-concurrency", 2,
        "--prefill-chunk", 128,
        "--no-prefix-reuse",
        "--host-context-mib", 512,
    ),
    "shared-growth-recovery": _args(
        "--max-context", 2688,
        "--kv-capacity", 2688,
        "--max-concurrency", 3,
        "--prefill-chunk", 128,
        "--device-state-slots", 2,
        "--host-context-mib", 0,
    ),
    "host-history-pressure": _args(
        "--max-context", 512,
        "--kv-capacity", 768,
        "--max-concurrency", 3,
        "--prefill-chunk", 128,
        "--device-state-slots", 0,
        "--host-context-mib", 384,
    ),
    "vision-growth-replay": _args(
        "--max-context", 1024,
        "--kv-capacity", 1024,
        "--max-concurrency", 2,
        "--prefill-chunk", 128,
        "--device-state-slots", 0,
        "--host-context-mib", 0,
        "--no-prefix-reuse",
        "--vision",
        "--media-cache-mib", 0,
        "--media-live-mib", 64,
    ),
}


for _capacity, _kv_tokens in (("roomy", 10240), ("pressure", 8192)):
    PROFILE_ARGS[f"resident-{_capacity}"] = _args(
        "--max-context", 8192, "--kv-capacity", _kv_tokens,
        "--max-concurrency", 3, "--prefill-chunk", 1024,
        "--no-prefix-reuse", "--host-context-mib", 0,
    )

PROFILE_ARGS["mixed-arrivals"] = _args(
    "--max-context", 8192, "--kv-capacity", 8192,
    "--max-concurrency", 4, "--prefill-chunk", 1024,
    "--max-pending-requests", 16, "--pending-timeout-ms", 120000,
    "--no-prefix-reuse", "--host-context-mib", 512,
)

for _route, _host_mib_value in (("replay", 0), ("snapshot", 512)):
    PROFILE_ARGS[f"agent-{_route}"] = _args(
        "--max-context", 768, "--kv-capacity", 768,
        "--max-concurrency", 2, "--prefill-chunk", 128,
        "--device-state-slots", 2, "--host-context-mib", _host_mib_value,
    )

for _backend, _draft_tokens in (("mtp", 3), ("dflash2", 7)):
    PROFILE_ARGS[f"preemption-snapshot-{_backend}"] = (
        *PROFILE_ARGS["preemption-snapshot"],
        "--spec", _backend, "--draft-tokens", str(_draft_tokens), "--lm-head-draft",
    )
