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

#include "resources/event.h"

#include "common.h"

namespace SparseOperationKit {

Event::Event(const std::string name) : name_(name) {
  CK_CUDA(hipEventCreateWithFlags(&cuda_event_, hipEventDisableTiming));
}

Event::~Event() {
  try {
    CK_CUDA(hipEventDestroy(cuda_event_));
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
  }
}

std::shared_ptr<Event> Event::create(const std::string name) {
  return std::shared_ptr<Event>(new Event(std::move(name)));
}

void Event::Record(hipStream_t& stream) {
  if (!IsReady())
    throw std::runtime_error(ErrorBase + "cudaEvent: " + name() + " is still in use.");
  CK_CUDA(hipEventRecord(cuda_event_, stream));
}

bool Event::IsReady() const {
  hipError_t error = hipEventQuery(cuda_event_);
  if (hipSuccess == error) {
    return true;
  } else if (hipErrorNotReady == error) {
    return false;
  } else {
    CK_CUDA(error);
    return false;
  }
}

void Event::TillReady(hipStream_t& stream) {
  CK_CUDA(hipStreamWaitEvent(stream, cuda_event_, 0));
}

void Event::TillReady() { CK_CUDA(hipEventSynchronize(cuda_event_)); }

std::string Event::name() const { return name_; }

}  // namespace SparseOperationKit