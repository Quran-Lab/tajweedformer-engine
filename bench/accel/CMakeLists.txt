# The conformance harnesses: each prints an FNV-1a fingerprint of every output of every chunk, which
# bench/v4_conformance_check.py compares with conformance/v4_*_expected.jsonl.
foreach(t zf_info_check zf_state_check zf_state_dump zf_stream_concurrent zf_stream_check
          zf_stream_pin_check zf_pin_default_check zf_vfs_layout_check)
  add_executable(${t} ${t}.c)
  target_compile_options(${t} PRIVATE -O2 -Wall -Wextra -Werror)
  target_link_libraries(${t} PRIVATE zf_stream)
endforeach()
# zf_set over the engine's graph runner (zf_rt), one session for a whole list of clips
add_executable(zf_set_rt zf_set.c)
target_compile_options(zf_set_rt PRIVATE -O2 -Wall -Wextra -Werror)
target_link_libraries(zf_set_rt PRIVATE zf_ort_ops zf_rt)
# the engine's files from a CoreML build and its sidecar (zf/zfs.h), for the Apple profile's checks
add_executable(zf_zfs_write zf_zfs_write.c)
target_compile_options(zf_zfs_write PRIVATE -O2 -Wall -Wextra -Werror)
target_link_libraries(zf_zfs_write PRIVATE zf_engine)
add_executable(zf_run_rt zf_run.c)
target_compile_options(zf_run_rt PRIVATE -O2 -Wall -Wextra -Werror)
target_link_libraries(zf_run_rt PRIVATE zf_ort_ops zf_rt)
# The V4 model folder through zf_tajweed.h (audio in, every head of every chunk fingerprinted).
if(TARGET zf_tajweed_inference AND (ANDROID OR APPLE OR UNIX OR MINGW))
  add_executable(zf_tajweed_check zf_tajweed_check.c)
  target_compile_options(zf_tajweed_check PRIVATE -O2 -Wall -Wextra -Werror)
  target_link_libraries(zf_tajweed_check PRIVATE zf_tajweed_inference)
endif()
