// A reproduction attempt for the port's issue 5, kept in its own file so the existing suite is
// unchanged.
//
// "Qwen3.5 resource subtraction underflow" kills the engine -- every in-flight request fails, /health
// answers 503 forever, and only a restart recovers. The reporter's incident analysis names the trigger
// precisely, and it is not cache pressure:
//
//   "A request that reuses the previous step's cached state (log shows `cache 9x % (private endpoint)`
//    on the preceding step) starts while a second request is still generating. The failure follows
//    0.1-1 s after that start."
//
// Their own table rules pressure out -- incidents occurred "with the sum of resident prompts far below
// --kv-capacity". So this case does not build pressure. It builds the *race*: a session with a reusable
// endpoint, a second lane generating while that endpoint's continuation is submitted, and the
// continuation carrying the previous turn's reasoning content so its prefix matches what was published.
//
// Why an engine test rather than HTTP, which several earlier attempts used. A cache hit requires
// `context_cache.session_key` and a retention hint, which the HTTP surface does not expose in the same
// form; every HTTP attempt reported `cache 0.0%` and therefore never crossed the trigger at all.
//
// Every step asserts and reports; a failure prints the capture counters, so a refusal names its own
// branch rather than being inferred.
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <utility>

#include "ninfer/engine.h"
#include "ninfer/types.h"

namespace {

// Sized so that (a) a capture is offered, (b) the source is eligible to be demoted once its request
// completes, and (c) demoting it is cheaper than rebuilding it. That last condition is what decides the
// pressure planner's choice: it prices a checkpoint's `rebuild_ns` against its
// `baseline_recovery_ns` (resource_manager.h:2115-2118) and only demotes when recovery costs less.
// A small session is trivially rebuildable, which is why nine earlier runs declined to demote at every
// pool size: runs 3-5 had large sessions against a large pool (nothing to demote for), and runs 7-9 had
// small sessions against a small pool (cheap to rebuild, so the planner preferred that).
//
// The reporter's sessions were 50k-230k tokens, where rebuilding is expensive. This keeps a session
// large while shrinking the pool, so both conditions hold at once. `kv >= context` is the engine's own
// invariant, and the 1.125 ratio is theirs.
constexpr std::uint32_t kContext     = 24576;
constexpr std::uint32_t kKvCapacity  = 27648;
constexpr std::uint32_t kPreChunk    = 2048;
constexpr std::uint32_t kConcurrency = 3;

// A session's size as a share of the pool, which is what decides whether anything is parked. The first
// run of this case used a 4,012-token root against a 65,536-token context -- about 6% full -- and
// nothing was demoted, so the restore path under test was unreachable and the clean result meant
// nothing. The reporter's own incidents run sessions at 19-78% of `--kv-capacity` (55k-230k against
// 294,912), so the root here is sized to that ratio and stated as a share rather than a word count.
constexpr double kRootPoolShare = 0.76;
constexpr std::uint32_t kRootWords =
    static_cast<std::uint32_t>((static_cast<double>(kContext) * kRootPoolShare) / 1.3);

ninfer::EngineOptions reporter_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path    = artifact;
    options.max_context      = kContext;
    options.kv_capacity      = ninfer::KvCapacityPolicy::explicit_capacity(kKvCapacity);
    options.prefill_chunk    = kPreChunk;
    // Their speculative backend: dflash2 with 7 draft tokens. Their table shows `--spec mtp` failing
    // differently, so the backend is not incidental to the trigger.
    options.speculative.backend                  = ninfer::SpeculativeBackend::DFlash2;
    options.speculative.draft_tokens             = 7;
    options.speculative.proposal_head            = ninfer::ProposalHead::Optimized;
    options.max_concurrency                      = kConcurrency;
    options.max_pending_requests                 = 32;
    options.context_cache.device_state_slots     = 2;
    options.context_cache.host_state_slots       = 16;
    options.context_cache.host_kv_capacity_bytes = std::uint64_t{6144} << 20;
    options.context_cache.max_private_continuations         = 8;
    options.context_cache.max_shared_prefixes               = 7;
    options.context_cache.max_long_anchors_per_continuation = 4;
    return options;
}

ninfer::RequestOptions request_options(std::uint32_t outputs, bool reuse) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = outputs;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = reuse;
    request.stop.include_model_defaults       = false;
    return request;
}

// A conversation root of the size their incidents involve. Their smallest observed pair is 52k + 81k
// tokens, and one incident happened "51 s after restart, no session above 63k on the engine" -- so a
// root near the low end of their range is the faithful choice rather than the largest.
std::string session_body(std::uint32_t words) {
    std::string text;
    text.reserve(6U * words);
    for (std::uint32_t index = 0; index < words; ++index) { text += "alpha "; }
    return text;
}

ninfer::PromptInput session_root(const std::string& body, const std::string& key,
                                 ninfer::CacheRetentionHint retention =
                                     ninfer::CacheRetentionHint::LiveSession) {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = body, .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking   = false;
    // The two fields that make a cache hit possible at all. Every earlier HTTP attempt lacked them,
    // which is why none of them ever reported a hit and none could have reached the failure.
    input.context_cache.session_key = key;
    input.context_cache.retention   = retention;
    // And the field that makes a *capture* possible, which is what a demotion requires. A cache
    // opportunity is built from the prompt's declared markers (request_plan.cpp:354 reads
    // `base->context_cache.opportunities`, and the gate at :326 requires `allow_prefix_reuse &&
    // prompt.identity.reusable && context_cache.enabled`). Without a marker no CaptureGroup is formed,
    // no capture is offered, and the counter that shows it is `active_captures_offered` -- which every
    // earlier run of this case reported as 0, so the pressure planner never ran and nothing could be
    // demoted to host.
    input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
        .after_message_count = 1,
        .kind                = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .location            = ninfer::PromptCacheMarkerLocation::MessageBoundary,
    });
    return input;
}

void report(const char* label, const ninfer::RuntimeStats& stats) {
    std::cerr << label
              << " state_h2d=" << stats.state_h2d_count
              << " main_h2d=" << stats.main_kv_h2d_pages
              << " backend_h2d=" << stats.backend_kv_h2d_pages
              << " degraded=" << stats.pressure_private_owners_degraded
              << " evicted=" << stats.pressure_private_owners_evicted
              << " captures[offered=" << stats.active_captures_offered
              << " no_vacancy=" << stats.active_captures_no_vacancy
              << " plan_refused=" << stats.active_captures_plan_refused
              << " infeasible=" << stats.active_captures_infeasible
              << " completed=" << stats.active_captures_completed
              << " aborted=" << stats.active_captures_aborted << "]\n";
}

// The case: a cache-hitting continuation submitted while another lane is still generating.
int exercise_concurrent_cache_hit(const char* artifact) {
    ninfer::Engine engine(reporter_engine_options(artifact));

    // 1. A session publishes a reusable private endpoint.
    const std::string body = session_body(kRootWords);
    const ninfer::GenerationResult source =
        engine.generate(engine.prepare(session_root(body, "issue5-race-a")),
                        request_options(16, true));
    if (source.generated_token_ids.size() != 16) {
        std::cerr << "session root did not complete: tokens=" << source.prompt.prompt_tokens
                  << " outputs=" << source.generated_token_ids.size() << '\n';
        return 1;
    }
    std::cout << "root published: prompt=" << source.prompt.prompt_tokens
              << " path=" << static_cast<int>(source.prefix_reuse_path)
              << " reused=" << source.reused_prompt_tokens << '\n';

    // The continuation of that session: same prefix, the assistant turn echoed back *including* its
    // reasoning content, then a new user line. That is what their client sends, and what makes the
    // frontier land where the previous request ended.
    ninfer::PromptInput continuation = session_root(body, "issue5-race-a");
    ninfer::ChatMessage assistant;
    assistant.role              = ninfer::ChatRole::Assistant;
    assistant.reasoning_content = source.reasoning;
    assistant.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = source.content, .media = {}});
    continuation.messages.push_back(std::move(assistant));
    ninfer::ChatMessage followup;
    followup.role = ninfer::ChatRole::User;
    followup.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "Continue with one short line.", .media = {}});
    continuation.messages.push_back(std::move(followup));

    // 2. Demote the source's turn closure to host, now that the source's request has completed.
    //
    //    The rule that reaches `PrefixReusePath::PrivateTurnClosure` fires only when a pressure action
    //    demoted the *closure's* State (resource_manager.h:408-427): while the closure is `DeviceOnly`
    //    the rule skips it and the session endpoint wins, which is why earlier runs reported
    //    `path=1 reused=38,340` with `state_h2d=0`.
    //
    //    The harder condition, and the one eight earlier runs missed, is what makes an owner *eligible*
    //    to be demoted at all. Every pressure-candidate filter -- resource_manager.h:738, 840, 955 and
    //    2079 -- skips any slot for which `private_has_active_edge(slot)` holds, and that predicate is
    //    (1389-1395):
    //
    //        any active lane whose `retained_private_source->slot == slot`
    //
    //    So an owner cannot be demoted while a lane is still holding it as its retained source. The
    //    existing `exercise_host_restore` satisfies this by being strictly sequential with
    //    `max_concurrency = 1`: its source request finishes and releases its edge before the pressure
    //    request runs. This case submits a holder early, so a lane is always occupied -- which is why no
    //    run demoted anything.
    //
    //    The source above has completed (its result was consumed by `generate`), so its edge is released
    //    and its slot is eligible. The pressure request is a distinct, non-reusing prompt, as the
    //    reference scenario uses.
    const ninfer::RuntimeStats before_pressure = engine.runtime_stats();
    // Pressure with a distinct, non-reusing prompt, exactly as `exercise_host_restore` does. Two earlier
    // attempts failed here for opposite reasons: one used a different session key with reuse disabled but
    // an oversized pool, and another reused the *same* key, which adds a private owner to the session
    // instead of competing with the existing one. The reference scenario pressures with a fresh prompt
    // and `reuse=false`, leaving the source's checkpoint as the only catalogued private owner -- which is
    // why the planner picks it as the victim.
    ninfer::PromptInput pressure = session_root(session_body(kRootWords), "issue5-race-pressure");
    const ninfer::GenerationResult pressure_result =
        engine.generate(engine.prepare(std::move(pressure)), request_options(2, false));
    const ninfer::RuntimeStats after_pressure = engine.runtime_stats();
    std::cout << "pressure: outputs=" << pressure_result.generated_token_ids.size() << '\n';
    report("  after pressure", after_pressure);
    // Reported, not required. An earlier version failed the run when nothing was demoted, which made the
    // loop assert a proximate condition of my own rather than the reporter's symptom -- the skill's Phase
    // 1 says a loop must be able to go red on *this* bug, and "did it demote" is not that bug. The value
    // is printed so the state is visible without gating the verdict on it.
    if (after_pressure.main_kv_d2h_pages <= before_pressure.main_kv_d2h_pages) {
        std::cout << "note: nothing was demoted by the pressure request (informational)\n";
    }

    // 3. The trigger: a restore racing an in-flight generation. The holder is submitted first and is
    //    still generating when the continuation starts, which is their reported timing -- the failure
    //    follows 0.1-1 s after a cache-hitting continuation starts against a busy engine.
    const std::string other_body = session_body(kRootWords);
    ninfer::GenerationHandle holder =
        engine.submit(engine.prepare(session_root(other_body, "issue5-race-b")),
                      request_options(512, false));
    std::cout << "holder submitted; generating while the continuation starts\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));

    const ninfer::RuntimeStats before = engine.runtime_stats();
    ninfer::GenerationHandle hitter =
        engine.submit(engine.prepare(std::move(continuation)), request_options(16, true));

    // 4. Collect. Either may throw; an underflow arrives as a RequestError from the engine, and any
    //    exception is reported with the counters rather than swallowed.
    bool underflow = false;
    std::string failure;
    try {
        const ninfer::GenerationResult held = holder.wait();
        std::cout << "holder finished: outputs=" << held.generated_token_ids.size() << '\n';
    } catch (const std::exception& error) {
        failure = error.what();
    }
    try {
        const ninfer::GenerationResult hit = hitter.wait();
        std::cout << "continuation finished: reused=" << hit.reused_prompt_tokens
                  << " path=" << static_cast<int>(hit.prefix_reuse_path)
                  << " outputs=" << hit.generated_token_ids.size() << '\n';
        // Reported, not required, for the same reason as the demotion check above. Whether reuse took the
        // endpoint or the closure path is a proximate condition; the reporter's symptom is the underflow
        // and the permanent unavailability, and that is what this loop must be able to catch.
        if (hit.reused_prompt_tokens == 0) {
            std::cout << "note: the continuation reused nothing (informational)\n";
        }
    } catch (const std::exception& error) {
        failure = error.what();
    }

    const ninfer::RuntimeStats after = engine.runtime_stats();
    report("  after", after);
    if (after.state_h2d_count > before.state_h2d_count) {
        std::cout << "note: the continuation restored state from host, which is the path under test\n";
    }

    if (!failure.empty()) {
        std::cerr << "a request failed: " << failure << '\n';
        underflow = failure.find("underflow") != std::string::npos ||
                    failure.find("subtraction") != std::string::npos;
    }
    if (underflow) {
        std::cerr << "RESOURCE SUBTRACTION UNDERFLOW REPRODUCED\n";
        return 1;
    }
    // A latched engine would refuse further work; probe it, because the reporter's signature is that
    // the engine stays unavailable rather than recovering.
    try {
        const ninfer::GenerationResult probe =
            engine.generate(engine.prepare(session_root("alpha ", "issue5-race-probe")),
                            request_options(4, false));
        if (probe.generated_token_ids.size() != 4) {
            std::cerr << "engine did not complete a follow-up request after the race\n";
            return 1;
        }
        std::cout << "engine still accepts work after the race (not latched)\n";
    } catch (const std::exception& error) {
        std::cerr << "engine refused a follow-up request: " << error.what() << '\n';
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        return exercise_concurrent_cache_hit(artifact);
    } catch (const std::exception& error) {
        std::cerr << "unhandled: " << error.what() << '\n';
        return 1;
    }
}
