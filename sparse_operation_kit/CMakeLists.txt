# 
# Copyright (c) 2021, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.8)
project(SparseOperationKit LANGUAGES CXX)

message(STATUS "Building Sparse Operation Kit from source.")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_HIP_STANDARD 14)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmakes)
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

find_package(hip REQUIRED)
find_package(RCCL REQUIRED)
find_package(TensorFlow REQUIRED)
find_package(MPI REQUIRED)
find_package(Threads)

#set(HIP_SEPARABLE_COMPILATION ON)

# whether use nvtx
option(USE_NVTX "Use nvtx for profiling" OFF)
if (USE_NVTX)
    message(STATUS "Add nvtx for profiling")
    add_definitions(-DUSE_NVTX)
    find_package(NVTX REQUIRED)
endif()

# whether make ops to be asynchronous
option(SOK_ASYNC "Make Ops to be asynchronous" ON)
if (SOK_ASYNC)
    message(STATUS "Make Ops to be asynchronous")
    add_definitions(-DSOK_ASYNC)
endif()

# setting compiler flags
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Setting default CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
endif()

foreach(arch_name ${SM})
    if (arch_name STREQUAL gfx908)
        list(APPEND rocm_arch_list ${arch_name})
        message(STATUS "Assign GPU architecture (sm=${arch_name})")
    else()
        message(FATAL_ERROR "Unknown or unsupported GPU architecture (set sm=gfx908)")
    endif()
endforeach()

list(LENGTH rocm_arch_list rocm_arch_list_length)
if(${rocm_arch_list_length} EQUAL 0)
    list(APPEND rocm_arch_list "gfx908")
    message(STATUS "Assign default GPU architecture sm=gfx908")
endif()
list(REMOVE_DUPLICATES rocm_arch_list)

foreach(arch_name ${rocm_arch_list})
    set(GPU_TARGETS "${arch_name}" CACHE STRING "GPU targets to compile for")
endforeach()

if (${CMAKE_BUILD_TYPE} EQUAL "Release")
    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS} -Wall -Werror")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-unknown-pragmas")
    if (${TF_VERSION_MAJOR} EQUAL 1)
        set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-error=class-memaccess")
    endif()
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -rdc=true -Xcompiler -Wall,-Werror,-Wno-error=cpp")
    # set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Xcudafe --display_error_number")
    # set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
else() 
    # -------- set flags for DEBUG mode --- #
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wno-unknown-pragmas")
    if (${TF_VERSION_MAJOR} EQUAL 1)
        set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wno-error=class-memaccess")
    endif()
    set(CMAKE_HIP_FLAGS_DEBUG "${CMAKE_HIP_FLAGS_DEBUG} -rdc=true -Xcompiler -Wall,-Wno-error=cpp")
    # set(CMAKE_HIP_FLAGS_DEBUG "${CMAKE_HIP_FLAGS_DEBUG} -Xcudafe --display_error_number")
    # set(CMAKE_HIP_FLAGS_DEBUG "${CMAKE_HIP_FLAGS_DEBUG} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
    # ------------------------------------- #
endif()

# set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

set(CMAKE_CXX_FLAGS "${CMAKE_HIP_FLAGS} ${TF_COMPILE_FLAGS} -DTENSORFLOW_USE_ROCM=1 /opt/rocm/lib/libhiprand.so /opt/rocm/lib/librccl.so /opt/rocm/lib/libhipsparse.so -DHIP_PLATFORM=AMD -DEIGEN_USE_HIP")
# set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} ${TF_COMPILE_FLAGS} -DTENSORFLOW_USE_ROCM=1 -DHIP_PLATFORM=AMD -DEIGEN_USE_HIP")

# setting output folder
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# setting install folder
install(DIRECTORY ${CMAKE_BINARY_DIR}/lib DESTINATION /usr/local)

# headers
include_directories(
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/include/
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/third_party/
    ${PROJECT_SOURCE_DIR}/kit_cc/framework/kernels/
    ${PROJECT_SOURCE_DIR}/kit_cc_impl/embedding/
    ${HIP_INCLUDE_DIRS}
    ${RCCL_INC_PATHS}
    ${MPI_INCLUDE_PATH}
)


# libs
link_directories(
    ${RCCL_LIB_PATHS}
    ${TF_LINK_DIR}/
)

# code sources related to framework
file(GLOB framework
    ${PROJECT_SOURCE_DIR}/kit_cc/framework/kernels/*.cpp
    ${PROJECT_SOURCE_DIR}/kit_cc/framework/kernels/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/framework/kernels/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc/framework/ops/*.cpp
    ${PROJECT_SOURCE_DIR}/kit_cc/framework/ops/*.cc
)

# code sources related to the kit infrastructure
file(GLOB kit_infra
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/*.cpp
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/resources/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/parameters/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/parameters/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/hashtable/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/hashtable/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/embeddings/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/embeddings/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/tensor_buffer/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/dispatcher/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/dispatcher/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/initializer/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/initializer/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/operation/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/operation/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/optimizer/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/optimizer/*.cu
)

# code sources for the concrete algorithm implementations of the kit
file(GLOB kit_impl
    ${PROJECT_SOURCE_DIR}/kit_cc_impl/embedding/common/src/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc_impl/embedding/common/src/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc_impl/embedding/dispatcher/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc_impl/embedding/dispatcher/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc_impl/embedding/lookuper/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc_impl/embedding/lookuper/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc_impl/embedding/operation/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc_impl/embedding/operation/*.cu
)

file(GLOB test_op
    ${PROJECT_SOURCE_DIR}/Plugin/framework/kernels/test_op.cc
    ${PROJECT_SOURCE_DIR}/Plugin/framework/ops/test_op.cc
)

# build dynamic lib
add_library(sparse_operation_kit SHARED ${framework} ${kit_infra} ${kit_impl})
target_link_libraries(sparse_operation_kit PUBLIC ${TF_LINK_FLAGS} ${MPI_CXX_LIBRARIES})

if (USE_NVTX)
    target_link_libraries(sparse_operation_kit PUBLIC ${NVTX_LIB})
endif()

set_target_properties(sparse_operation_kit PROPERTIES HIP_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(sparse_operation_kit PROPERTIES HIP_ARCHITECTURES OFF)

# compile compat ops
add_subdirectory(kit_cc_impl/operations)

# compile ops for unit test
option(SOK_UNIT_TEST "Compile SOK unit test" ON)
if (SOK_UNIT_TEST)
    message(STATUS "Compile SOK unit test")
    add_subdirectory(unit_test)
endif()