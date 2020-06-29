#ifndef TVM_RUNTIME_CONTRIB_POPLAR_FN_INFO_H_
#define TVM_RUNTIME_CONTRIB_POPLAR_FN_INFO_H_

#include <dlpack/dlpack.h>

#include <vector>
#include <string>

namespace tvm {
namespace runtime {
namespace contrib {

struct PoplarFunctionInfo {
  public:
  int program_index;
  std::vector<DLDataType> arg_types;
  std::vector<std::string> input_channels;
  std::string output_channel;
};

}
}
}

#endif
