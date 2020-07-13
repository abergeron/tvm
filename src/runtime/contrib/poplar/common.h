#ifndef TVM_RUNTIME_CONTRIB_POPLAR_COMMON_H_
#define TVM_RUNTIME_CONTRIB_POPLAR_COMMON_H_

#include <dmlc/thread_local.h>
#include <tvm/runtime/device_api.h>

#include <poplar/Device.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>

namespace tvm {
namespace runtime {
namespace contrib {

class IPUThreadEntry {
 public:
  IPUThreadEntry() : valid_(false), active_engine_(nullptr) {}

  static IPUThreadEntry* ThreadLocal() { return dmlc::ThreadLocalStore<IPUThreadEntry>::Get(); }

  void set_device(poplar::Device&& dev) {
    if (valid_) device_.detach();
    device_ = std::move(dev);
    device_.attach();
    valid_ = true;
    active_engine_ = nullptr;
  }

  void set_active_engine(poplar::Engine* eng) {
    CHECK(valid_) << "Set an engine on an invalid device";
    if (eng != active_engine_) {
      eng->load(device_);
      active_engine_ = eng;
    }
  }

 private:
  bool valid_;
  poplar::Device device_;
  poplar::Engine* active_engine_;
};

class IPUDeviceAPI final : public DeviceAPI {
 public:
  IPUDeviceAPI() : m_(poplar::DeviceManager::createDeviceManager()) {
    // XXX: Hard-code this for now, don't use from multiple threads
    IPUThreadEntry* t = GetThreadEntry();
    bool use_model = false;
    char* tmp = getenv("TVM_POPLAR_USE_MODEL");

    if (tmp != NULL) use_model = bool(atoi(tmp));

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

#endif
