#include <dmlc/thread_local.h>

#include <tvm/runtime/registry.h>

#include "common.h"

namespace tvm {
namespace runtime {
namespace contrib {

void IPUDeviceAPI::GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) {
  size_t index = static_cast<size_t>(ctx.device_id);
  if (kind == kExist) {
    *rv = static_cast<int>(index < m_.getNumDevices());
    return;
  }
  CHECK_LT(index, m_.getNumDevices()) << "Invalid device id " << index;

  // None of the properties for IPU seem relevant for these so we just
  // fake them for now.

  switch(kind) {
    case kDeviceName: {
      *rv = std::string("IPU");
      break;
    }
    case kMaxClockRate: {
      *rv = 1300;
      break;
    }
    case kMultiProcessorCount: {
      *rv = 1216;
      break;
    }
    default:
      return;
  }
}

void* IPUDeviceAPI::AllocDataSpace(TVMContext ctx, size_t nbytes, size_t alignment,
				   DLDataType type_hint) {
  // We allocate the buffers on the CPU since it's not really possible
  // to allocate on the device. We ignore alignment since this is not
  // the location that will be referenced for execution.
  void* ptr;
  ptr = malloc(nbytes);
  if (ptr == nullptr) throw std::bad_alloc();
  return ptr;
}

void IPUDeviceAPI::FreeDataSpace(TVMContext ctx, void* ptr)  {
  free(ptr);
}

void IPUDeviceAPI::CopyDataFromTo(const void* from, size_t from_offset,
				  void* to, size_t to_offset, size_t size,
				  TVMContext ctx_from, TVMContext ctx_to,
				  DLDataType type_hint, TVMStreamHandle stream) {
  memcpy(static_cast<char*>(to) + to_offset, static_cast<const char*>(from) + from_offset, size);
}


TVM_REGISTER_GLOBAL("device_api.ipu").set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = IPUDeviceAPI::Global().get();
    *rv = static_cast<void*>(ptr);
});

}
}
}
