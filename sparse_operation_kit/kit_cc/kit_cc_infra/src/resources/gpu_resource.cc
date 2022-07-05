/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "resources/gpu_resource.h"

#include <cstdlib>
#include <iostream>

#include "common.h"
#ifdef USE_NVTX
#include <nvToolsExtCudaRt.h>
#endif

namespace SparseOperationKit {

bool GetEventSync() {
  const auto sok_event_sync = std::getenv("SOK_EVENT_SYNC");
  if (nullptr == sok_event_sync || 1 == std::atoi(sok_event_sync)) {
    return true;
  } else {
    return false;
  }
}

GpuResource::GpuResource(const size_t local_device_id, const size_t global_device_id,
                         const uint64_t replica_uniform_seed, const uint64_t replica_variant_seed,
                         const ncclComm_t& nccl_comm, const hipStream_t& hip_stream)
    : local_device_id_(local_device_id),
      global_device_id_(global_device_id),
      computation_stream_(nullptr),
      framework_stream_(hip_stream),
      nccl_comm_(nccl_comm),
      sm_count_(0),
      cc_major_(0),
      cc_minor_(0),
      max_shared_memory_size_per_sm_(0),
      warp_size_(0),
      nccl_sync_data_(nullptr),
      event_mgr_(nullptr),
      event_sync_(GetEventSync()) {
#ifdef SOK_ASYNC
  CK_CUDA(hipStreamCreateWithFlags(&computation_stream_, hipStreamNonBlocking));
  event_mgr_.reset(EventManager::create().release());
#ifdef USE_NVTX
  nvtxNameCudaStreamA(computation_stream_, "SOKComputStream");
#endif
#else
#ifdef USE_NVTX
  nvtxNameCudaStreamA(framework_stream_, "FrameworkComputStream");
#endif
  computation_stream_ =
      framework_stream_;  // sok will use the same hipStream_t created by framework.
#endif
  CK_CURAND(hiprandCreateGenerator(&replica_uniform_curand_generator_, HIPRAND_RNG_PSEUDO_DEFAULT));
  CK_CURAND(
      hiprandSetPseudoRandomGeneratorSeed(replica_uniform_curand_generator_, replica_uniform_seed));
  CK_CURAND(hiprandSetStream(replica_uniform_curand_generator_, computation_stream_));
  CK_CURAND(hiprandCreateGenerator(&replica_variant_curand_generator_, HIPRAND_RNG_PSEUDO_DEFAULT));
  CK_CURAND(
      hiprandSetPseudoRandomGeneratorSeed(replica_variant_curand_generator_, replica_variant_seed));
  CK_CURAND(hiprandSetStream(replica_variant_curand_generator_, computation_stream_));

  CK_CUSPARSE(hipsparseCreate(&replica_cusparse_handle_));
  CK_CUSPARSE(hipsparseSetStream(replica_cusparse_handle_, computation_stream_));

  CK_CUDA(hipDeviceGetAttribute(&sm_count_, hipDeviceAttributeMultiprocessorCount, local_device_id_));
  CK_CUDA(hipDeviceGetAttribute(&cc_major_, hipDeviceAttributeComputeCapabilityMajor, local_device_id_));
  CK_CUDA(hipDeviceGetAttribute(&cc_minor_, hipDeviceAttributeComputeCapabilityMinor, local_device_id_));
  CK_CUDA(hipDeviceGetAttribute(&max_shared_memory_size_per_sm_,
                                 hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, local_device_id_));
  max_shared_memory_size_per_sm_ -= (4 * 1024ul);  // FIXME: in case it allocates all shared memory.
  CK_CUDA(hipDeviceGetAttribute(&warp_size_, hipDeviceAttributeWarpSize, local_device_id_));

  CK_CUDA(hipMalloc(&nccl_sync_data_, sizeof(int32_t) * 1));
}

GpuResource::~GpuResource() {
  try {
    CK_NCCL(ncclCommDestroy(nccl_comm_));
    CK_CURAND(hiprandDestroyGenerator(replica_uniform_curand_generator_));
    CK_CURAND(hiprandDestroyGenerator(replica_variant_curand_generator_));
    CK_CUSPARSE(hipsparseDestroy(replica_cusparse_handle_));
#ifdef SOK_ASYNC
    CK_CUDA(hipStreamDestroy(computation_stream_));
#endif
    if (nccl_sync_data_) CK_CUDA(hipFree(nccl_sync_data_));
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
  }
}

std::shared_ptr<GpuResource> GpuResource::Create(const size_t local_device_id,
                                                 const size_t global_device_id,
                                                 const uint64_t replica_uniform_seed,
                                                 const uint64_t replica_variant_seed,
                                                 const ncclComm_t& nccl_comm,
                                                 const hipStream_t& hip_stream) {
  return std::shared_ptr<GpuResource>(new GpuResource(local_device_id, global_device_id,
                                                      replica_uniform_seed, replica_variant_seed,
                                                      nccl_comm, hip_stream));
}

size_t GpuResource::get_local_device_id() const { return local_device_id_; }

size_t GpuResource::get_global_device_id() const { return global_device_id_; }

hipStream_t& GpuResource::get_stream() { return computation_stream_; }

hipStream_t& GpuResource::get_framework_stream() { return framework_stream_; }

size_t GpuResource::get_sm_count() const { return static_cast<size_t>(sm_count_); }

size_t GpuResource::get_max_smem_size_per_sm() const {
  return static_cast<size_t>(max_shared_memory_size_per_sm_);
}

size_t GpuResource::get_warp_size() const { return static_cast<size_t>(warp_size_); }

const hiprandGenerator_t& GpuResource::get_variant_curand_gen() const {
  return replica_variant_curand_generator_;
}

const hiprandGenerator_t& GpuResource::get_uniform_curand_gen() const {
  return replica_uniform_curand_generator_;
}

const ncclComm_t& GpuResource::get_nccl() const { return nccl_comm_; }

const hipsparseHandle_t& GpuResource::get_cusparse() const { return replica_cusparse_handle_; }

void GpuResource::sync_gpu_via_nccl(const hipStream_t& stream) const {
  CK_NCCL(ncclAllReduce(/*sendbuff=*/nccl_sync_data_,
                        /*recvbuff=*/nccl_sync_data_,
                        /*count=*/1ul,
                        /*datatype=*/ncclUint64,
                        /*op=*/ncclMax, nccl_comm_, stream));
}

void GpuResource::event_record(EventRecordType event_record_type, const std::string event_name) {
  switch (event_record_type) {
    case EventRecordType::RDLFramework: {
      event_mgr_->sync_two_streams(/*root_stream=*/get_framework_stream(),
                                   /*sub_stream=*/get_stream(),
                                   /*event_name=*/std::move(event_name),
                                   /*event_sync=*/event_sync_);
      break;
    }
    case EventRecordType::RMyself: {
      event_mgr_->sync_two_streams(/*root_stream=*/get_stream(),
                                   /*sub_stream=*/get_framework_stream(),
                                   /*event_name=*/std::move(event_name));
      break;
    }
    default: {
      throw std::runtime_error(ErrorBase + "Not supported EventRecordType.");
      break;
    }
  }  // switch block
}

}  // namespace SparseOperationKit
