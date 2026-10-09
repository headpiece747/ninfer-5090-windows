add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
ninfer_internal_includes(ninfer)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
ninfer_internal_includes(ninfer-serve)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_logging)

add_executable(ninfer-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp
  perplexity/topk_record.cpp)
ninfer_internal_includes(ninfer-perplexity)
target_link_libraries(ninfer-perplexity PRIVATE
  ninfer_engine ninfer_product_logging ninfer::json)

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
