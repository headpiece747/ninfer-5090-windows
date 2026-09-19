#include "runtime/engine/context_cache/materialization_budget.h"

#include <iostream>
#include <stdexcept>

using namespace ninfer::runtime;

static void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

int main() {
    constexpr std::uint64_t ms = 1'000'000;
    try {
        auto idle = PlanningAllowance::boundary(0, 0);
        // The initial grant is min(250 ms, incumbent_cost / 20, allowance). These scenarios were
        // written against the 5 ms flat cap the grant used to be pinned at, so a 100 ms incumbent
        // reproduces exactly that grant (100/20 = 5 ms) and every assertion below keeps its
        // original meaning. The scaled policy itself is pinned further down.
        MaterializationSearchBudget expensive(idle, 0, 100 * ms);
        require(expensive.allow(4 * ms, 2 * ms, 3 * ms, 70'000 * ms, true, 1),
                "valuable completion could not cross the initial window");
        require(expensive.granted_ns() == 10 * ms && expensive.renewals() == 1,
                "extension did not retain cumulative accounting");
        require(!expensive.allow(9 * ms, 2 * ms, 2 * ms, ms, true, 2),
                "improved incumbent retained the expensive root's budget");
        require(expensive.stop_reason() ==
                    ninfer::MaterializationStopReason::InsufficientExpectedGain,
                "economic stopping was reported as wall exhaustion");

        MaterializationSearchBudget discovery(idle, 0, 100 * ms);
        require(discovery.allow(5 * ms, ms, 4 * ms, 70'000 * ms, false, 1),
                "unknown candidate could not receive bounded discovery");
        require(!discovery.allow(10 * ms, ms, ms, 70'000 * ms, false, 2),
                "unknown candidate repeatedly renewed discovery");
        require(discovery.allow(10 * ms, ms, ms, 70'000 * ms, true, 2),
                "complete prediction could not continue after discovery");

        auto busy = PlanningAllowance::boundary(2, 0);
        MaterializationSearchBudget first(busy, busy.limit_ns - 4 * ms, 80'000 * ms);
        require(first.granted_ns() == 4 * ms, "mandatory work did not consume boundary time");
        MaterializationSearchBudget backfill(busy, busy.limit_ns - ms, 80'000 * ms);
        require(backfill.granted_ns() == ms, "backfill reset the boundary allowance");
        require(!backfill.allow(busy.limit_ns, 1, 1, 70'000 * ms, true, 1),
                "search delayed runnable requests beyond their boundary");
        require(backfill.overshoot(busy.limit_ns + ms) == ms,
                "indivisible overrun was not measured");

        // The concurrent 55K rotation spends ~3.5 ms preparing/validating identities. A useful
        // complete target must still be assessable; discarding the whole cache makes all six
        // subsequent continuations cold. The old 5 ms boundary left only 1.5 ms and failed here.
        MaterializationSearchBudget restore_after_setup(busy, 3 * ms + ms / 2, 80'000 * ms);
        require(restore_after_setup.allow(3 * ms + ms / 2, 2 * ms, 2 * ms, 70'000 * ms, true, 1),
                "mandatory setup starved the first complete reuse assessment in a busy boundary");

        MaterializationSearchBudget cheap(idle, 0, ms);
        require(cheap.granted_ns() == ms / 20, "cheap request received a minimum 5 ms grant");
        require(!cheap.allow(ms / 20, ms, ms, ms, false, 1),
                "discovery ignored the economic cap for a cheap request");
        MaterializationSearchBudget saturated(idle, 0, UINT64_MAX);
        require(saturated.granted_ns() == 0 && !saturated.allow(0, ms, ms, UINT64_MAX, true, 1),
                "saturated cost was used as evidence of unlimited gain");
        MaterializationSearchBudget seeded(idle, 0, 100 * ms);
        require(!seeded.allow(5 * ms, ms, ms, 70'000 * ms, false, 1, false),
                "an already-seeded candidate renewed solely on an incomplete optimistic estimate");
        require(seeded.allow(5 * ms, ms, ms, 70'000 * ms, true, 1, false),
                "a complete profitable refinement was denied after seeding");
        std::atomic<bool> cancelled{false};
        auto controlled                = PlanningAllowance::boundary(0, 0);
        controlled.cancellation        = &cancelled;
        controlled.control_deadline_ns = 3 * ms;
        require(controlled.remaining(2 * ms) == ms, "control deadline did not constrain planning");
        cancelled.store(true);
        require(controlled.remaining(0) == 0, "cancelled request kept optional planning headroom");
        MaterializationSearchBudget stalled(idle, 0, 100 * ms);
        require(stalled.allow(5 * ms, ms, ms, 70'000 * ms, true, 7), "first forecast grant failed");
        require(!stalled.allow(10 * ms, ms, ms, 70'000 * ms, true, 7),
                "stalled work renewed its allowance");
        // The initial grant scales with the incumbent's cost, so a search that has a genuinely
        // better plan to find is not cut off before it verifies that plan. A flat 5 ms cap left a
        // reuse target unverified on a busy engine and admission then fell back to the root
        // incumbent, re-prefilling the whole prompt (upstream issue #229, TTFT 142 s -> 1.2 s once
        // the grant scales). The economic term still governs below the ceiling, and the boundary
        // allowance still caps both.
        const PlanningAllowance roomy{
            .started_ns = 0, .limit_ns = 10'000 * ms, .affected_requests = 1};
        MaterializationSearchBudget scaled(roomy, 0, 80'000 * ms);
        require(scaled.granted_ns() == 250 * ms, "a large incumbent did not scale the search grant");
        MaterializationSearchBudget governed(roomy, 0, 4'000 * ms);
        require(governed.granted_ns() == 200 * ms,
                "the economic term did not govern below the ceiling");
        MaterializationSearchBudget allowance_bounded(busy, 0, 80'000 * ms);
        require(allowance_bounded.granted_ns() == busy.limit_ns,
                "the boundary allowance no longer capped the scaled grant");
        std::cout << "ok\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
