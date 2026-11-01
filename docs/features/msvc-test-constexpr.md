# MSVC test portability

This branch makes existing C++20 tests compile without compiler-specific extensions.
It replaces test-local `constexpr` expressions involving `std::sqrt` with `const`;
the represented scale formula and oracle checks are unchanged.
It also includes `<array>` directly where tests use `std::array` deduction.
No production kernel, runtime option or numerical tolerance is changed.

Configure the test tree with `BUILD_TESTING=ON` and run it through `tools/scripts/test_v3.cmd`, which
builds `build-test/` and runs the affected attention, GDN replay and KV-cache tests with CTest. Do not
use the apps tree (`build/`), which is configured with `BUILD_TESTING` off.
This is a test portability prerequisite, not the complete Windows platform port.
See the [test guide](../../tests/README.md) and [build contract](../maintainer/build-system.md).
