#pragma once

#include <cstdint>

namespace ninfer {

#if defined(__SIZEOF_INT128__)

// Every compiler that provides a native 128-bit integer uses it, so the runtime contract and the
// context-cost arithmetic stay character-for-character the original on GCC and Clang (including
// Clang on Windows). Only MSVC, which has no __int128, needs the fallback below.
using Uint128 = unsigned __int128;

#else

// MSVC has no __int128. This is a minimal unsigned 128-bit integer covering exactly the saturating
// cost-model arithmetic the runtime contract and context-cost code need: a little-endian 64-bit
// pair with wrap-around semantics on overflow identical to the GCC/Clang builtin type.
//
// Every operation is constexpr, so upstream's constant expressions (for example the `maximum`
// sentinel in PrefillWork) remain constant expressions on MSVC instead of having to be relaxed.
// The arithmetic is written in 32-bit limbs rather than MSVC's _umul128/_addcarry_u64 intrinsics
// precisely because the intrinsics are not constexpr.
struct Uint128 {
    std::uint64_t lo = 0;
    std::uint64_t hi = 0;

    constexpr Uint128() = default;
    constexpr Uint128(std::uint64_t value) noexcept : lo(value), hi(0) {}
    constexpr Uint128(std::uint64_t low, std::uint64_t high) noexcept : lo(low), hi(high) {}

    constexpr explicit operator std::uint64_t() const noexcept { return lo; }

    // High word of the 64x64 -> 128 product, in 32-bit limbs. This is the exact quantity MSVC's
    // non-constexpr _umul128 reports through its out-parameter. The inner sum is at most
    // 3*(2^32-1) so it never wraps, which is what makes the single carry shift below sufficient.
    [[nodiscard]] static constexpr std::uint64_t widening_multiply_high(std::uint64_t a,
                                                                      std::uint64_t b) noexcept {
        const std::uint64_t a_low  = a & 0xFFFFFFFFULL;
        const std::uint64_t a_high = a >> 32U;
        const std::uint64_t b_low  = b & 0xFFFFFFFFULL;
        const std::uint64_t b_high = b >> 32U;
        const std::uint64_t low_low    = a_low * b_low;
        const std::uint64_t low_high   = a_low * b_high;
        const std::uint64_t high_low   = a_high * b_low;
        const std::uint64_t high_high  = a_high * b_high;
        const std::uint64_t middle =
            (low_low >> 32U) + (low_high & 0xFFFFFFFFULL) + (high_low & 0xFFFFFFFFULL);
        return high_high + (low_high >> 32U) + (high_low >> 32U) + (middle >> 32U);
    }
};

[[nodiscard]] constexpr Uint128 operator*(const Uint128& a, const Uint128& b) noexcept {
    // Same three terms the intrinsic version accumulated: the low word of the low product, its
    // high word, and the low words of the two cross products. Everything else is a multiple of
    // 2^128 and vanishes under truncation, exactly as it does in the builtin type.
    const std::uint64_t low  = a.lo * b.lo;
    const std::uint64_t high = Uint128::widening_multiply_high(a.lo, b.lo) + (a.lo * b.hi) +
                               (a.hi * b.lo);
    return Uint128{low, high};
}

[[nodiscard]] constexpr Uint128 operator+(const Uint128& a, const Uint128& b) noexcept {
    const std::uint64_t low   = a.lo + b.lo;
    const std::uint64_t carry = low < a.lo ? 1ULL : 0ULL;
    return Uint128{low, a.hi + b.hi + carry};
}

[[nodiscard]] constexpr Uint128 operator-(const Uint128& a, const Uint128& b) noexcept {
    const std::uint64_t low    = a.lo - b.lo;
    const std::uint64_t borrow = a.lo < b.lo ? 1ULL : 0ULL;
    return Uint128{low, a.hi - b.hi - borrow};
}

[[nodiscard]] constexpr Uint128 operator~(const Uint128& value) noexcept {
    return Uint128{~value.lo, ~value.hi};
}

[[nodiscard]] constexpr Uint128 operator<<(const Uint128& value, unsigned shift) noexcept {
    if (shift == 0) { return value; }
    if (shift >= 128) { return Uint128{}; }
    if (shift >= 64) { return Uint128{0, value.lo << (shift - 64)}; }
    return Uint128{value.lo << shift, (value.hi << shift) | (value.lo >> (64 - shift))};
}

[[nodiscard]] constexpr Uint128 operator>>(const Uint128& value, unsigned shift) noexcept {
    if (shift == 0) { return value; }
    if (shift >= 128) { return Uint128{}; }
    if (shift >= 64) { return Uint128{value.hi >> (shift - 64), 0}; }
    return Uint128{(value.lo >> shift) | (value.hi << (64 - shift)), value.hi >> shift};
}

// Long division by a 64-bit divisor; the cost models only ever divide by small constants, so the
// bit-serial loop is fast enough and, unlike a hardware intrinsic, stays constexpr.
[[nodiscard]] constexpr Uint128 operator/(const Uint128& value, std::uint64_t divisor) noexcept {
    Uint128 quotient{};
    std::uint64_t rem_hi = 0;
    std::uint64_t rem_lo = 0;
    for (int bit = 127; bit >= 0; --bit) {
        const std::uint64_t word       = bit >= 64 ? value.hi : value.lo;
        const unsigned bit_in_word     = static_cast<unsigned>(bit & 63);
        const std::uint64_t bit_value  = (word >> bit_in_word) & 1ULL;
        const std::uint64_t shifted_lo = (rem_lo << 1) | bit_value;
        rem_hi = (rem_hi << 1) | (rem_lo >> 63);
        rem_lo = shifted_lo;
        // Loop invariant: the remainder stays below 2*divisor <= 2^65, so at most one subtraction
        // per step, and a high word of 1 always clears.
        if (rem_hi != 0 || rem_lo >= divisor) {
            rem_lo = rem_lo - divisor; // modular: also correct when rem_hi == 1
            rem_hi = 0;
            if (bit >= 64) {
                quotient.hi |= (1ULL << (bit - 64));
            } else {
                quotient.lo |= (1ULL << bit);
            }
        }
    }
    return quotient;
}

// Remainder against a 64-bit divisor, matching the builtin's `unsigned __int128 % uint64_t`.
// serve/serve_options.cpp needs it to reject --host-context-mib values that do not resolve to a whole
// number of bytes, which is upstream's parse (b9114396) written against the native type. Derived
// rather than repeated: a % b is a - (a / b) * b, so the long-division loop above is the only place
// that needs to be correct.
[[nodiscard]] constexpr Uint128 operator%(const Uint128& value, std::uint64_t divisor) noexcept {
    return value - operator/(value, divisor) * Uint128{divisor};
}

// Uint128 / Uint128, because upstream's --host-context-mib parse divides two values it built as
// 128-bit (a MiB numerator against a power-of-ten divisor), and the builtin it was written against
// accepts that. Splitting a 128-bit divisor needs the same shift-and-subtract walk, with a wider
// remainder: the running remainder stays below 2 * divisor, so it spans three limbs.
[[nodiscard]] constexpr Uint128 operator/(const Uint128& value, const Uint128& divisor) noexcept {
    if (divisor.hi == 0) { return operator/(value, divisor.lo); }
    Uint128 quotient{};
    std::uint64_t rem[3] = {0, 0, 0}; // little-endian: rem[0] is least significant
    for (int bit = 127; bit >= 0; --bit) {
        const std::uint64_t word      = bit >= 64 ? value.hi : value.lo;
        const unsigned bit_in_word    = static_cast<unsigned>(bit & 63);
        const std::uint64_t bit_value = (word >> bit_in_word) & 1ULL;
        // rem = rem * 2 + bit, across three limbs.
        const std::uint64_t carry = rem[2] >> 63;
        rem[2]                    = (rem[2] << 1) | (rem[1] >> 63);
        rem[1]                    = (rem[1] << 1) | (rem[0] >> 63);
        rem[0]                    = (rem[0] << 1) | bit_value;
        (void)carry;
        // Loop invariant: rem < 2 * divisor, so one conditional subtraction clears it.
        // rem is three limbs and the divisor two, so the divisor's high limb lines up with rem[1],
        // not rem[2]: the alignment is rem[2]:rem[1]:rem[0] against 0:divisor.hi:divisor.lo.
        // Comparing rem[2] against divisor.hi instead is off by 64 bits, which made every dividend
        // below 2^128 look larger than a 10^20 divisor and returned a zero quotient.
        bool ge = (rem[2] != 0) || (rem[1] > divisor.hi) ||
                  (rem[1] == divisor.hi && rem[0] >= divisor.lo);
        bool borrow = false;
        if (ge) {
            std::uint64_t r0 = rem[0], r1 = rem[1], r2 = rem[2];
            const std::uint64_t s0 = r0 - divisor.lo;
            borrow                = r0 < divisor.lo;
            const std::uint64_t b1 = r1 - divisor.hi - (borrow ? 1U : 0U);
            borrow                = borrow ? (r1 <= divisor.hi) : (r1 < divisor.hi);
            r0                    = s0;
            r1                    = b1;
            r2                    = r2 - (borrow ? 1U : 0U);
            rem[0]                = r0;
            rem[1]                = r1;
            rem[2]                = r2;
        }
        if (ge) {
            if (bit >= 64) {
                quotient.hi |= (1ULL << (bit - 64));
            } else {
                quotient.lo |= (1ULL << bit);
            }
        }
    }
    return quotient;
}

[[nodiscard]] constexpr Uint128 operator%(const Uint128& value, const Uint128& divisor) noexcept {
    return value - operator/(value, divisor) * divisor;
}

[[nodiscard]] constexpr bool operator==(const Uint128& a, const Uint128& b) noexcept {
    return a.hi == b.hi && a.lo == b.lo;
}
[[nodiscard]] constexpr bool operator!=(const Uint128& a, const Uint128& b) noexcept {
    return !(a == b);
}
[[nodiscard]] constexpr bool operator<(const Uint128& a, const Uint128& b) noexcept {
    return a.hi != b.hi ? a.hi < b.hi : a.lo < b.lo;
}
[[nodiscard]] constexpr bool operator>(const Uint128& a, const Uint128& b) noexcept {
    return b < a;
}
[[nodiscard]] constexpr bool operator<=(const Uint128& a, const Uint128& b) noexcept {
    return !(b < a);
}
[[nodiscard]] constexpr bool operator>=(const Uint128& a, const Uint128& b) noexcept {
    return !(a < b);
}

constexpr Uint128& operator*=(Uint128& a, const Uint128& b) noexcept { return a = a * b; }
constexpr Uint128& operator+=(Uint128& a, const Uint128& b) noexcept { return a = a + b; }
constexpr Uint128& operator-=(Uint128& a, const Uint128& b) noexcept { return a = a - b; }

#endif

} // namespace ninfer
