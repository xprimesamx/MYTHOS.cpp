cmake_minimum_required(VERSION 3.20)
project(MYTHOS.cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(arch)
include(compiler)

option(OIL_BUILD_TESTS "Build unit tests" ON)
option(OIL_BUILD_BENCHMARKS "Build benchmarks" ON)
option(OIL_BUILD_TOOLS "Build CLI tools" ON)
option(OIL_SANITIZE "Enable address sanitizer" OFF)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/oil_config.h.in"
  "${CMAKE_BINARY_DIR}/oil_config.h"
)

add_library(oil_config INTERFACE)
target_include_directories(oil_config INTERFACE "${CMAKE_BINARY_DIR}")

file(GLOB_RECURSE OIL_HEADERS "include/oil/*.h")
file(GLOB_RECURSE OIL_SOURCES "src/*.cpp")

add_library(oil_core
  src/tensor.cpp
  src/memory.cpp
  src/random.cpp
)
target_include_directories(oil_core PUBLIC include)
target_link_libraries(oil_core PUBLIC oil_config)

add_library(oil_math
  src/math.cpp
  src/math_avx2.cpp
)
target_include_directories(oil_math PUBLIC include)
target_link_libraries(oil_math PUBLIC oil_core)

add_library(oil_format
  src/oil_format.cpp
  src/codebook.cpp
  src/format_planner.cpp
)
target_include_directories(oil_format PUBLIC include)
target_link_libraries(oil_format PUBLIC oil_math)

add_library(oil_kernel
  src/kernel_i2s.cpp
  src/kernel_tl.cpp
  src/kernel_oil8.cpp
  src/kernel_oil4.cpp
  src/int8_quant.cpp
)
target_include_directories(oil_kernel PUBLIC include)
target_link_libraries(oil_kernel PUBLIC oil_format)

add_library(oil_model
  src/transformer.cpp
  src/model.cpp
  src/kv_cache.cpp
)
target_include_directories(oil_model PUBLIC include)
target_link_libraries(oil_model PUBLIC oil_kernel)

add_library(oil_inference
  src/sampler.cpp
  src/generator.cpp
)
target_include_directories(oil_inference PUBLIC include)
target_link_libraries(oil_inference PUBLIC oil_model)

add_library(oil_tokenizer
  src/bpe_tokenizer.cpp
)
target_include_directories(oil_tokenizer PUBLIC include)
target_link_libraries(oil_tokenizer PUBLIC oil_core)

add_library(oil_trainer
  src/autograd.cpp
  src/optimizer.cpp
  src/trainer.cpp
  src/ste_quantizer.cpp
  src/finetune.cpp
)
target_include_directories(oil_trainer PUBLIC include)
target_link_libraries(oil_trainer PUBLIC oil_model oil_inference oil_tokenizer)

add_library(oil_engine engines/inference/inference.cpp engines/inference/stream.cpp)
target_include_directories(oil_engine PUBLIC include engines/inference)
target_link_libraries(oil_engine PUBLIC oil_inference oil_tokenizer)

add_library(oil_oil8 engines/OIL8/codec.cpp engines/OIL8/quantize.cpp)
target_include_directories(oil_oil8 PUBLIC include engines/OIL8)
target_link_libraries(oil_oil8 PUBLIC oil_format)

add_library(oil_dense
  engines/trainer/dense/trainer.cpp
  engines/trainer/dense/dataloader.cpp
  engines/trainer/dense/checkpoint.cpp
)
target_include_directories(oil_dense PUBLIC include engines/trainer/dense)
target_link_libraries(oil_dense PUBLIC oil_trainer)

add_library(oil_moe
  engines/trainer/moe/moe.cpp
  engines/trainer/moe/vision/vision.cpp
)
target_include_directories(oil_moe PUBLIC include
  engines/trainer/moe
  engines/trainer/moe/vision
  engines/trainer/moe/audio
  engines/trainer/moe/embeddings
  engines/trainer/moe/image
  engines/trainer/moe/ocr
  engines/trainer/moe/text
  engines/trainer/moe/video
)
target_link_libraries(oil_moe PUBLIC oil_model)

add_library(oil_multimodal
  engines/trainer/multimodel/text/multimodal_text.cpp
  engines/trainer/multimodel/image/vision.cpp
  engines/trainer/multimodel/audio/audio.cpp
  engines/trainer/multimodel/video/video.cpp
  engines/trainer/multimodel/ocr/ocr.cpp
  engines/trainer/multimodel/embeddings/embeddings.cpp
  engines/trainer/multimodel/vision/vision.cpp
)
target_include_directories(oil_multimodal PUBLIC include
  engines/trainer/multimodel/text
  engines/trainer/multimodel/image
  engines/trainer/multimodel/audio
  engines/trainer/multimodel/video
  engines/trainer/multimodel/ocr
  engines/trainer/multimodel/embeddings
  engines/trainer/multimodel/vision
)
target_link_libraries(oil_multimodal PUBLIC oil_model)

if(OIL_BUILD_TOOLS)
  add_executable(oil_infer tools/infer.cpp)
  target_link_libraries(oil_infer PRIVATE oil_engine oil_tokenizer)

  add_executable(oil_train tools/train.cpp)
  target_link_libraries(oil_train PRIVATE oil_dense oil_tokenizer)

  add_executable(oil_finetune tools/finetune.cpp)
  target_link_libraries(oil_finetune PRIVATE oil_trainer)

  add_executable(oil_convert tools/convert.cpp)
  target_link_libraries(oil_convert PRIVATE oil_format)

  add_executable(oil_bench tools/bench.cpp)
  target_link_libraries(oil_bench PRIVATE oil_kernel oil_inference oil_tokenizer)

  add_executable(oil_info tools/info.cpp)
  target_link_libraries(oil_info PRIVATE oil_format)
endif()

if(OIL_BUILD_TESTS)
  enable_testing()
  macro(add_oil_test name src deps)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE include)
    target_link_libraries(${name} PRIVATE ${deps})
    add_test(NAME ${name} COMMAND ${name})
  endmacro()
  add_oil_test(test_tensor   tests/test_tensor.cpp   oil_core)
  add_oil_test(test_math     tests/test_math.cpp     oil_math)
  add_oil_test(test_format   tests/test_format.cpp   oil_format)
  add_oil_test(test_kernel   tests/test_kernel.cpp   oil_kernel)
  add_oil_test(test_model    tests/test_model.cpp     oil_model)
  add_oil_test(test_tokenizer tests/test_tokenizer.cpp oil_tokenizer)
  add_oil_test(test_trainer  tests/test_trainer.cpp  oil_trainer oil_inference oil_tokenizer)
  add_oil_test(test_debug    tests/test_debug.cpp    oil_model)
endif()

if(OIL_BUILD_BENCHMARKS)
  macro(add_oil_bench name src deps)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE include)
    target_link_libraries(${name} PRIVATE ${deps})
  endmacro()
  add_oil_bench(bench_kernels   bench/bench_kernels.cpp   oil_kernel)
  add_oil_bench(bench_inference bench/bench_inference.cpp oil_inference)
  add_oil_bench(bench_quality   bench/bench_quality.cpp   oil_inference)
endif()
