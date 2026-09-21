#pragma once

// Shared vocabulary for the row-split MMA route catalogs.
//
// kAnyCols is the sentinel a catalog's last route ends at: "this route covers every larger column
// count". Nine files defined it, every one as the same expression, which is the kind of fact that
// drifts one file at a time.
//
// The closure predicate belongs here too, once its callers agree on a form. Nine files carry one
// today, in four signature shapes, and they are all the same proof: routes_are_closed requires
// routes.back().last == kAnyCols && expected == kAnyCols + 1, catalog_is_closed requires only the
// second, and after the loop expected is last.last + 1 -- so the second implies the first. The two
// are equivalent, and the extra conjunct is worse than redundant, because it evaluates
// routes.back(), which is undefined for an empty array.

#include <cstdint>
#include <limits>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

} // namespace ninfer::ops::detail
