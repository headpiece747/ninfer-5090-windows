add_executable(ninfer cli/main.cpp)
ninfer_internal_includes(ninfer)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_config
  ninfer_product_logging
  ninfer_product_prompt_input)

if(NINFER_ENABLE_DISTRIBUTED)
  add_executable(ninfer-worker worker/main.cpp)
  ninfer_internal_includes(ninfer-worker)
  target_link_libraries(ninfer-worker PRIVATE ninfer_engine ninfer_runtime_distributed)
endif()

add_executable(ninfer-serve serve/main.cpp)
ninfer_internal_includes(ninfer-serve)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_logging)
if(WIN32)
  # timeBeginPeriod/timeEndPeriod live in winmm: the serve app raises the process timer resolution so
  # the engine's 1 ms scheduling poll means 1 ms rather than the 15.625 ms system default.
  target_link_libraries(ninfer-serve PRIVATE winmm)
endif()

add_executable(ninfer-perplexity
  perplexity/main.cpp
  perplexity/options.cpp
  perplexity/report.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp)
ninfer_internal_includes(ninfer-perplexity)
target_link_libraries(ninfer-perplexity PRIVATE
  ninfer_engine ninfer_product_config ninfer_product_logging ninfer::json)

# The three apps embed a UTF-8 process code page, because none of them sets a console code page and
# all of them write UTF-8 through printf/std::cout, so their output was correct only on a console
# already configured for it. See cmake/windows-utf8.manifest for the element, its namespace and what
# it does and does not guarantee.
if(MSVC)
  foreach(app ninfer ninfer-serve ninfer-perplexity)
    target_link_options(${app} PRIVATE
      "/MANIFESTINPUT:${CMAKE_SOURCE_DIR}/cmake/windows-utf8.manifest")
  endforeach()
endif()
