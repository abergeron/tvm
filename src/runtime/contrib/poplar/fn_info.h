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
