#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {
namespace contrib {

class IPUDeviceAPI final : public DeviceAPI {
public:
  void SetDevice(TVMContext ctx) final { }
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue *rv) final { }
  void *AllocDataSpace(TVMContext ctx, size_t nbytes, size_t alignment,
		       DLDataType type_hint) final { return NULL; }
  void FreeDataSpace(TVMContext ctx, void *ptr) final { }
  void CopyDataFromTo(const void* from, size_t from_offset,
		      void* to, size_t to_offset, size_t size,
		      TVMContext ctx_from, TVMContext ctx_to,
		      DLDataType type_hint, TVMStreamHandle stream) final { }
  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {}
  void* AllocWorkspace(TVMContext ctx, size_t size, DLDataType type_hint) final { return NULL; }
  void FreeWorkspace(TVMContext ctx, void* data) final { }

  static const std::shared_ptr<IPUDeviceAPI>& Global() {
    static std::shared_ptr<IPUDeviceAPI> inst = std::make_shared<IPUDeviceAPI>();
    return inst;
  }
};

TVM_REGISTER_GLOBAL("device_api.ipu").set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = IPUDeviceAPI::Global().get();
    *rv = static_cast<void*>(ptr);
});

}
}
}
