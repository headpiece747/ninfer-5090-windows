// plan_transfer: the half of materialization that touches neither the device nor the filesystem.
//
// These rules had no test surface before this split. The only route to them was materialize(),
// which allocates device memory and starts a transfer first -- something a machine with no device
// cannot do. On MSVC the injected-failure build cannot stand in for it either: GNU --wrap is
// unavailable there, and the import-address-table mechanism tests/artifact/tests.cmake mentions
// appears nowhere in the test source, so the CUDA failure cases print SKIP on the platform this
// port ships.
//
// What is checked here is the arithmetic only: overlap rejection, span coalescing, and slot sizing.
#include "artifact/materializer.h"

#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (!condition) {
        ++failures;
        std::printf("  FAIL  %s\n", label.c_str());
    }
}

bool rejected(const std::function<void()>& body) {
    try {
        body();
    } catch (const ninfer::artifact::ArtifactError&) {
        return true;
    }
    return false;
}

ninfer::artifact::CopyRange span(std::size_t file, std::uint64_t begin, std::uint64_t bytes) {
    return ninfer::artifact::CopyRange{file, begin, begin + bytes, nullptr};
}

} // namespace

int main() {
    using namespace ninfer::artifact;

    // Nothing to plan is not the same as planning zero bytes: an empty set produces no spans.
    expect(plan_transfer({}).spans.empty(), "empty ranges produce no spans");

    // Ranges in one file that a single aligned read can cover coalesce into one span.
    {
        const auto plan = plan_transfer({span(0, 8192, 4096), span(0, 12288, 4096)});
        expect(plan.spans.size() == 1, "adjacent ranges in one file coalesce");
        expect(plan.spans[0].begin == 8192 && plan.spans[0].end == 16384,
               "the coalesced span covers both ranges");
    }

    // Different files never share a span, however close their offsets are.
    {
        const auto plan = plan_transfer({span(0, 8192, 4096), span(1, 12288, 4096)});
        expect(plan.spans.size() == 2, "ranges in different files stay apart");
    }

    // A gap wider than the payload alignment starts a new span rather than reading across it.
    {
        const auto plan = plan_transfer({span(0, 8192, 4096), span(0, 65536, 4096)});
        expect(plan.spans.size() == 2, "a gap wider than the alignment starts a new span");
    }

    // Overlapping sources in one file are rejected: both would write the same destination bytes.
    expect(rejected([] { (void)plan_transfer({span(0, 8192, 8192), span(0, 12288, 4096)}); }),
           "overlapping source ranges are rejected");

    // The sorted ranges come back, because the transfer loop walks them in step with the spans.
    {
        const auto plan = plan_transfer({span(1, 4096, 4096), span(0, 4096, 4096)});
        expect(plan.ranges.size() == 2 && plan.ranges[0].file == 0 && plan.ranges[1].file == 1,
               "ranges come back sorted by file and then offset");
    }

    // Slot sizing: one slot when the payload fits, and never more than the cap.
    {
        const auto small = plan_transfer({span(0, 0, 4096)});
        expect(small.slot_count == 1, "a small payload needs one slot");
        expect(small.slot_bytes == small.aligned_bytes, "a small payload sizes the slot to fit");

        const auto large = plan_transfer({span(0, 0, 512ULL * 1024 * 1024)});
        expect(large.slot_count == 4, "a large payload is capped at four slots");
        expect(large.slot_bytes == 64ULL * 1024 * 1024, "a large payload sizes slots at 64 MiB");
    }

    if (failures == 0) { std::printf("  transfer plan checks passed\n"); }
    return failures == 0 ? 0 : 1;
}
