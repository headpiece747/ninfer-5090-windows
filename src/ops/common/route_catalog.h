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

#include <array>
#include <cstdint>
#include <limits>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

// The closure proof every catalog needs: the routes are contiguous from column 1, and the last one
// reaches kAnyCols, so no admitted column count can fall outside the catalog.
//
// Templated on the route type because the catalogs are: RouteSpec and Q8PairRouteSpec both carry
// first and last, and the proof reads nothing else. It takes the routes rather than closing over
// one, because three callers used to close over a file-local catalog -- one of them has moved to
// this form and the two that read a nested cols member deliberately have not -- and closing over a
// local is what made those three a different shape from the rest.
template <class Route, std::size_t N>
constexpr bool routes_are_closed(const std::array<Route, N>& routes) {
    std::int64_t expected = 1;
    for (const Route& route : routes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

} // namespace ninfer::ops::detail
