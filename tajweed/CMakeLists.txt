# The tajweed library (include/zf_tajweed.h): calibration, the expected table, the layered config and
# the scorer, in portable C11 with no dependencies. It builds inside the SDK (added from the top
# CMakeLists.txt) or on its own (cmake -S tajweed) for the scorer alone.
cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(zf_tajweed C)
  set(CMAKE_C_STANDARD 11)
  # every multiply-add rounds twice on every target (the scorer's bits must not depend on FMA)
  if(NOT MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-ffp-contract=off>)
  endif()
endif()
enable_testing()

set(ZF_TAJWEED_CORE src/names.c src/tj_json.c src/calib.c src/expected.c src/model.c src/config.c src/tj_ctc.c
    src/score.c src/tj_libm.c src/tj_sha256.c)
# The engine glue: engine_stub.c (no streams) unless a build names the real glue's sources.
set(ZF_TAJWEED_ENGINE_GLUE "${CMAKE_CURRENT_SOURCE_DIR}/src/engine_stub.c" CACHE STRING
    "Sources that implement zf_tajweed__engine_open/close and the zf_tajweed_stream_* functions")

if(MSVC)
  set(ZF_TAJWEED_WARN /W4)
else()
  set(ZF_TAJWEED_WARN -O2 -Wall -Wextra -Werror)
  if(WIN32)
    # MinGW ignores the header's visibility attribute with a warning
    list(APPEND ZF_TAJWEED_WARN -Wno-attributes)
  endif()
endif()

# The scorer spreads an ayah's sites over a few threads (score.c, score_windows); without a thread
# library (MSVC, WebAssembly without pthreads) it scores on the calling thread, with the same bits.
find_package(Threads)
if(NOT Threads_FOUND OR MSVC)
  add_compile_definitions(ZF_TJ_NO_THREADS=1)
endif()
# MinGW: winpthreads linked in statically, so the DLL needs no libwinpthread-1.dll beside it
set(ZF_TJ_THREAD_LIBS Threads::Threads)
if(MINGW)
  set(ZF_TJ_THREAD_LIBS -Wl,-Bstatic -lpthread -Wl,-Bdynamic)
endif()

function(zf_tajweed_target name kind)
  add_library(${name} ${kind} ${ZF_TAJWEED_CORE} ${ZF_TAJWEED_ENGINE_GLUE})
  if(Threads_FOUND AND NOT MSVC)
    target_link_libraries(${name} PRIVATE ${ZF_TJ_THREAD_LIBS})
  endif()
  target_include_directories(${name} PUBLIC include PRIVATE src)
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:C>:${ZF_TAJWEED_WARN}>)
  set_target_properties(${name} PROPERTIES C_VISIBILITY_PRESET hidden)
  if(WIN32)
    # fopen / strcpy on bounded buffers: the Windows CRT's deprecation notes are not errors here
    target_compile_definitions(${name} PRIVATE _CRT_SECURE_NO_WARNINGS)
  endif()
  if(UNIX AND NOT APPLE)
    target_link_libraries(${name} PRIVATE m)
  endif()
endfunction()

# What the SDKs link.
zf_tajweed_target(zf_tajweed STATIC)
# vendored libm (optimized-routines / fdlibm): its own style
set_source_files_properties(src/tj_libm.c PROPERTIES COMPILE_OPTIONS "-w")

# What the SDKs link when they have the engine: the same library with the real streams.
if(TARGET zf_stream)
  add_library(zf_tajweed_engine STATIC ${ZF_TAJWEED_CORE} src/engine.c)
  target_include_directories(zf_tajweed_engine PUBLIC include PRIVATE src)
  target_compile_options(zf_tajweed_engine PRIVATE $<$<COMPILE_LANGUAGE:C>:${ZF_TAJWEED_WARN}>)
  target_link_libraries(zf_tajweed_engine PUBLIC zf_stream)
  if(Threads_FOUND AND NOT MSVC)
    target_link_libraries(zf_tajweed_engine PRIVATE ${ZF_TJ_THREAD_LIBS})
  endif()
  set_source_files_properties(src/tj_libm.c PROPERTIES COMPILE_OPTIONS "-w")
  if(UNIX AND NOT APPLE)
    target_link_libraries(zf_tajweed_engine PRIVATE m)
  endif()
endif()

# For tests (tajweed/tests/run_vectors.py loads it with ctypes).
zf_tajweed_target(zf_tajweed_shared SHARED)
target_compile_definitions(zf_tajweed_shared PRIVATE ZF_TAJWEED_BUILD_DLL=1)
set_target_properties(zf_tajweed_shared PROPERTIES OUTPUT_NAME zf_tajweed_shared PREFIX "")

add_executable(zf_tajweed_test_config tests/test_config.c)
target_link_libraries(zf_tajweed_test_config PRIVATE zf_tajweed)
target_include_directories(zf_tajweed_test_config PRIVATE src)
add_test(NAME zf_tajweed_config COMMAND zf_tajweed_test_config ${CMAKE_CURRENT_SOURCE_DIR}/tests/data)
