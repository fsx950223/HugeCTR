/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#pragma once

#include "common.hpp"

namespace embedding {

class ILookup {
 public:
  virtual ~ILookup() = default;

  virtual void lookup(const Tensor &keys, size_t num_keys, const Tensor &id_space_offset,
                      size_t num_id_space_offset, const Tensor &id_space,
                      TensorList &embedding_vec) = 0;

  virtual void hash_insert(const Tensor &keys, size_t num_keys, const Tensor &offsets,
                           size_t num_offsets, const Tensor &d_id_space_list, Tensor &indices) {}

  virtual void scatter_lookup(const Tensor &indices, size_t num_indices, const Tensor &offsets,
                              size_t num_offsets, const Tensor &dst_idx, Tensor &emb_vec,
                              Tensor &emb_vec_offset) {}
};
}  // namespace embedding