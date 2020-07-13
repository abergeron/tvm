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

#ifndef TVM_RUNTIME_CONTRIB_POPLAR_COMMON_H_
#define TVM_RUNTIME_CONTRIB_POPLAR_COMMON_H_

#include <dmlc/thread_local.h>
#include <tvm/runtime/device_api.h>

#include <memory>
// clang-format and cpplint are fighting about the location of this header
#include <utility>

#include <poplar/Device.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>

namespace tvm {
namespace runtime {
namespace contrib {

// I'm not sure we need this, but it was inspired by the OpenCL code.

class IPUThreadEntry {
 public:
 IPUThreadEntry() : active_engine_(nullptr), valid_(false) {}

  static IPUThreadEntry* ThreadLocal() { return dmlc::ThreadLocalStore<IPUThreadEntry>::Get(); }

  /*!
   * \brief Set the current device
   * This can only be done by move since poplar::Device only supports that
   */
  void set_device(poplar::Device&& dev) {
    // We detach the previous device before since we don't want to hog
    // all the devices
    if (valid_) device_.detach();
    device_ = std::move(dev);
    // We need to attach to the device before doing anything useful.
    device_.attach();
    valid_ = true;
    active_engine_ = nullptr;
  }

  void set_active_engine(poplar::Engine* eng) {
    // Only one engine can be ative (loaded) on a device at a time.
    // If you loda a new one, the previous one is implicitely
    // replaced.  We keep track of which one is active for function
    // calls in order to avoid the reloading the engine every time
    // which can be expansive.
    CHECK(valid_) << "Set an engine on an invalid device";
    if (eng != active_engine_) {
      eng->load(device_);
      active_engine_ = eng;
    }
  }

 private:
  /*! The current device (might be the null device) */
  poplar::Device device_;
  /*! The current active engine (optional) */
  // We don't need to keep a hard reference to this because the only
  // use we have for it is to compare pointer values.
  poplar::Engine* active_engine_;
  /*! Is the current device a valid device (not the null one) */
  bool valid_;
};

// We don't need this API implementation for the current relay backend
// compiler use, but if we ever go into generating code for ops this
// might be needed.
class IPUDeviceAPI final : public DeviceAPI {
 public:
  IPUDeviceAPI() : m_(poplar::DeviceManager::createDeviceManager()) {
    // XXX: Hard-code this for now, don't use from multiple threads

    // This is bad, but I didn't want to take time to figure out how
    // to do it correctly.  We will need to remove all this code and
    // do it properly before a PR.
    IPUThreadEntry* t = GetThreadEntry();
    bool use_model = false;
    char* tmp = getenv("TVM_POPLAR_USE_MODEL");

    if (tmp != NULL) use_model = static_cast<bool>(atoi(tmp));

    if (!use_model) {
      t->set_device(m_.getDevice(0));
    } else {
      poplar::IPUModel m;
      t->set_device(m.createDevice());
    }
  }

  void SetDevice(TVMContext ctx) final {
    CHECK_LT(ctx.device_id, m_.getNumDevices()) << "Invalid device id " << ctx.device_id;
    IPUThreadEntry* t = GetThreadEntry();
    t->set_device(m_.getDevice(ctx.device_id));
  }
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final;
  void* AllocDataSpace(TVMContext ctx, size_t nbytes, size_t alignment, DLDataType type_hint) final;
  void FreeDataSpace(TVMContext ctx, void* ptr) final;
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      TVMContext ctx_from, TVMContext ctx_to, DLDataType type_hint,
                      TVMStreamHandle stream) final;

  // There is no support for streams
  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {}

  IPUThreadEntry* GetThreadEntry() { return IPUThreadEntry::ThreadLocal(); }

  static const std::shared_ptr<IPUDeviceAPI>& Global() {
    static std::shared_ptr<IPUDeviceAPI> inst = std::make_shared<IPUDeviceAPI>();
    return inst;
  }

 private:
  poplar::DeviceManager m_;
};

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_POPLAR_COMMON_H_
