#ifndef TVM_RUNTIME_CONTRIB_POPLAR_FN_INFO_H_
#define TVM_RUNTIME_CONTRIB_POPLAR_FN_INFO_H_

#include <dlpack/dlpack.h>
#include <dmlc/io.h>
#include <dmlc/json.h>

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

  void Save(dmlc::JSONWriter* writer) const;
  void Load(dmlc::JSONReader* reader);
  void Save(dmlc::Stream* writer) const;
  bool Load(dmlc::Stream* reader);
};

 using pop_fn_info = std::unordered_map<std::string, PoplarFunctionInfo>;

}
}
}

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, ::tvm::runtime::contrib::PoplarFunctionInfo, true);
}

#endif
