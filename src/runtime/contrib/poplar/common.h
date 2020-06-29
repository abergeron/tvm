#ifndef TVM_RUNTIME_CONTRIB_POPLAR_COMMON_H_
#define TVM_RUNTIME_CONTRIB_POPLAR_COMMON_H_

#include <dmlc/thread_local.h>

#include <tvm/runtime/device_api.h>

#include <poplar/Engine.hpp>
#include <poplar/Device.hpp>
#include <poplar/DeviceManager.hpp>

namespace tvm {
namespace runtime {
namespace contrib {

class IPUThreadEntry {
public:
 IPUThreadEntry() : active_engine_(nullptr) {}

  static IPUThreadEntry* ThreadLocal() {
    return dmlc::ThreadLocalStore<IPUThreadEntry>::Get();
  }

  bool valid() {
    return (&device_.getImpl() == nullptr);
  }

  void set_device(poplar::Device&& dev) {
    if (valid())
      device_.detach();
    device_ = std::move(dev);
    device_.attach();
    active_engine_ = nullptr;
  }

  void set_active_engine(poplar::Engine* eng) {
    CHECK(valid()) << "Set an engine on an invalid device";
    if (eng != active_engine_) {
      eng->load(device_);
      active_engine_ = eng;
    }
  }

 private:
  poplar::Device device_;
  poplar::Engine *active_engine_;
};

class IPUDeviceAPI final : public DeviceAPI {
public:
  IPUDeviceAPI() : m_(poplar::DeviceManager::createDeviceManager()) {}

  void SetDevice(TVMContext ctx) final {
    CHECK_LT(ctx.device_id, m_.getNumDevices()) << "Invalid device id " << ctx.device_id;
    IPUThreadEntry* t = GetThreadEntry();
    t->set_device(m_.getDevice(ctx.device_id));
  }
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final;
  void* AllocDataSpace(TVMContext ctx, size_t nbytes, size_t alignment,
                       DLDataType type_hint) final;
  void FreeDataSpace(TVMContext ctx, void* ptr) final;
  void CopyDataFromTo(const void* from, size_t from_offset,
                      void* to, size_t to_offset, size_t size,
                      TVMContext ctx_from, TVMContext ctx_to,
                      DLDataType type_hint, TVMStreamHandle stream) final;
  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {}

  IPUThreadEntry* GetThreadEntry() { return IPUThreadEntry::ThreadLocal(); }

  static const std::shared_ptr<IPUDeviceAPI>& Global() {
    static std::shared_ptr<IPUDeviceAPI> inst = std::make_shared<IPUDeviceAPI>();
    return inst;
  }
 private:
  poplar::DeviceManager m_;
};

}
}
}

#endif
