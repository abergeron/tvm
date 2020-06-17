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

namespace tvm {
namespace runtime {

class PoplarModule : public ModuleNode {
public:
  explicit PoplarModule();

  PackedFunc GetFunction(const std::string& name,
			 const ObjectPtr<Object>& sptr_to_self) final;

  const char* type_key() const { return "poplar"; }

  void SaveToBinary(dmlc::Stream* stream) final;
  static Module LoadFromBinary(void *strm);
  static Module Create(const std::string& path);
  std::string GetSource(const std::string& format = "");
  void Run(int id, const std::vector<int>& inputs, int output);
}
  
}
}
