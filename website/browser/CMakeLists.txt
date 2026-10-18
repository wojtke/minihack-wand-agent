# NLE WebAssembly build, modified 2026-10-17. See licenses/NetHack.txt.
cmake_minimum_required(VERSION 3.15)
project(wand_nle C CXX)
set(NLE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(NLE_PORT "${WAND_PORT}")
set(HACKDIR "/game")
set(INSTDIR "${CMAKE_BINARY_DIR}/install")
set(NLE_SRC ${NLE_ROOT}/src)
set(NLE_INC ${NLE_ROOT}/include)
set(NLE_DAT ${NLE_ROOT}/dat)
set(NLE_UTIL ${NLE_ROOT}/util)
set(NLE_DOC ${NLE_ROOT}/doc)
set(NLE_WIN ${NLE_ROOT}/win)
set(NLE_SRC_GEN ${CMAKE_BINARY_DIR}/src)
set(NLE_INC_GEN ${CMAKE_BINARY_DIR}/include)
set(NLE_DAT_GEN ${CMAKE_BINARY_DIR}/dat)
set(NLE_UTIL_GEN ${CMAKE_BINARY_DIR}/util)
add_compile_definitions(GCC_WARN NOCLIPPING NOMAIL NOTPARMDECL HACKDIR="/game"
  DEFAULT_WINDOW_SYS="rl" DLB NOCWD_ASSUMPTIONS NLE_USE_TILES NLE_ALLOW_SEEDING)
add_compile_options(-Wno-deprecated-non-prototype -Wno-error=implicit-function-declaration
  "-ffile-prefix-map=${NLE_ROOT}=nle" "-ffile-prefix-map=${NLE_PORT}=browser"
  "-ffile-prefix-map=${CMAKE_BINARY_DIR}=nle-build")
add_subdirectory(util)
foreach(tool makedefs dgn_comp lev_comp dlb recover tilemap)
  target_link_options(${tool} PRIVATE -sNODERAWFS=1 -sENVIRONMENT=node -sEXIT_RUNTIME=1)
endforeach()
add_subdirectory(dat)
file(GLOB GAME_SRC "src/*.c" "sys/share/posixregex.c" "sys/share/ioctl.c"
  "sys/unix/unixunix.c" "sys/unix/unixmain.c" "sys/unix/unixres.c" "win/tty/*.c" "win/rl/winrl.cc")
# Keep only the library sources, excluding standalone tools/tests.
set(BZ2_SRC third_party/bzip2/blocksort.c third_party/bzip2/bzlib.c
 third_party/bzip2/compress.c third_party/bzip2/crctable.c third_party/bzip2/decompress.c
 third_party/bzip2/huffman.c third_party/bzip2/randtable.c)
add_executable(wand ${GAME_SRC} ${BZ2_SRC} third_party/libtmt/tmt.c
 "${NLE_PORT}/fiber.c" "${NLE_PORT}/bridge.c")
add_dependencies(wand util dat)
set_target_properties(wand PROPERTIES CXX_STANDARD 14 SUFFIX ".js")
target_include_directories(wand PRIVATE ${NLE_INC} ${NLE_INC_GEN} ${NLE_PORT}
 third_party/libtmt third_party/bzip2)
target_link_options(wand PRIVATE
 -sASYNCIFY=1 -sALLOW_MEMORY_GROWTH=1 -sSTACK_SIZE=1048576 -sASYNCIFY_STACK_SIZE=262144
 -sMODULARIZE=1 -sEXPORT_NAME=createWandEngine -sENVIRONMENT=web,worker,node
 -sEXPORTED_RUNTIME_METHODS=ccall,FS,UTF8ToString,HEAPU8,HEAP16,HEAP32
 -sEXPORTED_FUNCTIONS=_wand_start,_wand_step,_wand_pointer,_wand_flag
 -sASSERTIONS=1 -sEXIT_RUNTIME=0 -sFORCE_FILESYSTEM=1)
