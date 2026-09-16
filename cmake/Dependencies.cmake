find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
if(MSVC)
  # Windows: FFmpeg dev binaries are staged into ffmpeg/ (see build_windows.bat).
  # Expose the same target name upstream consumers link, so nothing else changes.
  add_library(PkgConfig::FFMPEG INTERFACE IMPORTED GLOBAL)
  target_include_directories(PkgConfig::FFMPEG INTERFACE
    "${PROJECT_SOURCE_DIR}/ffmpeg/include")
  target_link_directories(PkgConfig::FFMPEG INTERFACE
    "${PROJECT_SOURCE_DIR}/ffmpeg/lib")
  target_link_libraries(PkgConfig::FFMPEG INTERFACE
    avcodec avformat avutil swscale swresample)
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil libswscale)
endif()

# Repository-pinned header dependencies. No configure-time downloads.
add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

# Source base for the custom-template frontend; consumers will link it explicitly.
add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)

if(NINFER_BUILD_PRODUCT_SUPPORT)
  # Media acquisition uses CURLOPT_PROTOCOLS_STR and CURLOPT_REDIR_PROTOCOLS_STR,
  # introduced in libcurl 7.85 (not merely the version of the maintainer environment).
  if(NOT MSVC)
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
  endif()
  add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  add_subdirectory(third_party/spdlog)
endif()
