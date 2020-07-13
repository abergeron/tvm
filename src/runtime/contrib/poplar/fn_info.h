/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef TVM_RUNTIME_CONTRIB_POPLAR_FN_INFO_H_
#define TVM_RUNTIME_CONTRIB_POPLAR_FN_INFO_H_

#include <dlpack/dlpack.h>
#include <dmlc/io.h>
#include <dmlc/json.h>

#include <string>
#include <vector>

namespace tvm {
namespace runtime {
namespace contrib {

struct PoplarFunctionInfo {
 public:
  int program_index;
  std::vector<std::string> input_channels;
  std::string output_channel;

  void Save(dmlc::JSONWriter* writer) const;
  void Load(dmlc::JSONReader* reader);
  void Save(dmlc::Stream* writer) const;
  bool Load(dmlc::Stream* reader);
};

using pop_fn_info = std::unordered_map<std::string, PoplarFunctionInfo>;

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, ::tvm::runtime::contrib::PoplarFunctionInfo, true);
}

#endif
