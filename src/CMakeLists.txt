llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-kv-cache-dsa.cpp
            llama-kv-cache-dsv4.cpp
            llama-loader-xnvme.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-hybrid-iswa.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampler.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

# xNVMe async weight loader: LLAMA_XNVME=ON auto-detects via pkg-config.
# If xnvme is not found, --loader xnvme falls back to STREAM at runtime.
option(LLAMA_XNVME "llama: link xNVMe for --loader xnvme" ON)
if (LLAMA_XNVME)
    find_package(PkgConfig QUIET)
    if (PkgConfig_FOUND)
        pkg_check_modules(XNVME IMPORTED_TARGET xnvme)
    endif()
    if (XNVME_FOUND)
        target_link_libraries(llama PRIVATE PkgConfig::XNVME)
        target_compile_definitions(llama PRIVATE LLAMA_USE_XNVME)
        message(STATUS "llama: xNVMe loader enabled (version ${XNVME_VERSION})")

        # upcie-cuda P2P path in llama-loader-xnvme.cpp calls cudaMemcpyAsync
        # / cudaEvent / cudaStream, which needs the CUDA runtime. When
        # GGML_CUDA is on the toolchain is already there; probe for it and
        # link the runtime library only into the llama target.
        if (GGML_CUDA)
            find_package(CUDAToolkit)
            if (CUDAToolkit_FOUND)
                target_link_libraries(llama PRIVATE CUDA::cudart)
                target_include_directories(llama PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
                message(STATUS "llama: xNVMe upcie-cuda P2P path enabled (CUDA runtime linked)")
            else()
                message(STATUS "llama: CUDAToolkit not found; --xnvme-be upcie-cuda will fall back")
            endif()
        endif()
    else()
        message(STATUS "llama: xNVMe not found via pkg-config; --loader xnvme will fall back to STREAM")
    endif()
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
