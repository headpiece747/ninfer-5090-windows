# The release tooling's checks. This tree has no Python test runner and does not need one: a
# script that exits non-zero is what ctest wants, which is how writer_interop.py is registered.
add_test(NAME ninfer_release_launcher_generation_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/test_launcher_generation.py")
