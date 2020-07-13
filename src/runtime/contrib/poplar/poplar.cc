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

#include <dmlc/logging.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <map>
#include <string>
#include <vector>

#include "../../pack_args.h"
#include "common.h"
#include "fn_info.h"

namespace tvm {
namespace runtime {
namespace contrib {

class PoplarWrappedFunc;

class PoplarModule : public ModuleNode {
 public:
  explicit PoplarModule(poplar::Executable&& exe,
                        const std::unordered_map<std::string, PoplarFunctionInfo>& fmap)
      : eng_(std::move(exe)), fmap_(fmap) {}

  const char* type_key() const { return "poplar"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;

  /* These are probably not possible for now */
  void SaveToBinary(dmlc::Stream* stream) final { CHECK(false) << "not possible"; }

  static Module LoadFromBinary(void* strm) {
    CHECK(false) << "not possible";
    return Module();
  }

  std::string GetSource(const std::string& format = "") {
    CHECK(false) << "not for now";
    return "";
  }

  void ensure_current() {
    IPUThreadEntry* t = IPUDeviceAPI::Global()->GetThreadEntry();
    // This is cached and will do nothing if engine is already loaded.
    t->set_active_engine(&eng_);
  }

 private:
  poplar::Engine eng_;
  std::unordered_map<std::string, PoplarFunctionInfo> fmap_;
  friend class PoplarWrappedFunc;
};

class PoplarWrappedFunc {
 public:
  PoplarWrappedFunc(PoplarModule* m, PoplarFunctionInfo& info, ObjectPtr<Object> sptr)
      : m_(m), info_(info), sptr_(sptr) {}

  void operator()(TVMArgs args, TVMRetValue* rv) const {
    m_->ensure_current();

    for (int i = 0; i < args.num_args; ++i) {
      TVM_CHECK_TYPE_CODE(args.type_codes[i], kTVMNDArrayHandle);
    }

    // Setup arguments
    int i = 0;
    for (const auto& it : info_.input_channels) {
      int index = i++;
      NDArray arg = args[index];
      m_->eng_.writeTensor(it, arg->data);
    }

    // run the function;
    m_->eng_.run(info_.program_index);

    // Get function result.
    NDArray ret = args[i];
    m_->eng_.readTensor(info_.output_channel, ret->data);
  }

 private:
  PoplarModule* m_;
  PoplarFunctionInfo& info_;

  // I don't know why we need this
  ObjectPtr<Object> sptr_;
};

PackedFunc PoplarModule::GetFunction(const std::string& name,
                                     const ObjectPtr<Object>& sptr_to_self) {
  if (name == "get_symbol") {
    return nullptr;
  } else if (name == "get_const_vars") {
    return nullptr;
  }

  const auto& it = fmap_.find(name);
  if (it == fmap_.end()) {
    for (auto it = fmap_.begin(); it != fmap_.end(); ++it)
      LOG(WARNING) << "available function: " << it->first;
    LOG(FATAL) << "Unknown function: " << name << "\n";
    return PackedFunc();
  }

  PoplarWrappedFunc f(this, it->second, sptr_to_self);
  return PackedFunc(f);
}

TVM_REGISTER_GLOBAL("module.poplar_module_create").set_body_typed([](void* exe_, void* fmap_) {
  // If there is a way to not go through void pointers, I would like
  // to know it.
  // Maybe if we dump/load the Executable, but still need to deal with
  // the function map (although that could be dump/loaded too maybe).
  auto* exe = static_cast<poplar::Executable*>(exe_);
  auto* fmap = static_cast<std::unordered_map<std::string, PoplarFunctionInfo>*>(fmap_);
  auto m = make_object<PoplarModule>(std::move(*exe), *fmap);
  return runtime::Module(m);
});

void PoplarFunctionInfo::Save(dmlc::JSONWriter* writer) const {
  writer->BeginObject();
  writer->WriteObjectKeyValue("program_index", program_index);
  writer->WriteObjectKeyValue("input_channels", input_channels);
  writer->WriteObjectKeyValue("output_channel", output_channel);
  writer->EndObject();
}

void PoplarFunctionInfo::Load(dmlc::JSONReader* reader) {
  dmlc::JSONObjectReadHelper helper;
  helper.DeclareField("program_index", &program_index);
  helper.DeclareField("input_channels", &input_channels);
  helper.DeclareField("output_channel", &output_channel);
  helper.ReadAllFields(reader);
}

void PoplarFunctionInfo::Save(dmlc::Stream* writer) const {
  writer->Write(program_index);
  writer->Write(input_channels);
  writer->Write(output_channel);
}

bool PoplarFunctionInfo::Load(dmlc::Stream* reader) {
  if (!reader->Read(&program_index)) return false;
  if (!reader->Read(&input_channels)) return false;
  if (!reader->Read(&output_channel)) return false;
  return true;
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
