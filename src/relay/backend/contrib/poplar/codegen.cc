#include <poplar/Graph.hpp>
#include <poplar/Target.hpp>
#include <poplar/Program.hpp>
#include <poplar/Type.hpp>
#include <poplar/VariableMappingMethod.hpp>
#include <poplar/Engine.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>
#include "../../utils.h"

#include "../../../../runtime/contrib/poplar/fn_info.h"
#include "../../../../runtime/contrib/poplar/common.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

using PoplarFunctionInfo = tvm::runtime::contrib::PoplarFunctionInfo;

static poplar::Type to_poplar_type(DataType& dt) {
  if (dt.lanes() != 1) {
    LOG(FATAL) << "Poplar doesn't support multi-lane data types (for now)\n";
  }
  switch (dt.code()) {
  case kDLInt:
    switch (dt.bits()) {
    case 32:
      return poplar::INT;
    case 16:
      return poplar::SHORT;
    case 8:
      return poplar::SIGNED_CHAR;
    }
    break;
  case kDLUInt:
    switch (dt.bits()) {
    case 32:
      return poplar::UNSIGNED_INT;
    case 16:
      return poplar::UNSIGNED_SHORT;
    case 8:
      return poplar::UNSIGNED_CHAR;
    case 1:
      return poplar::BOOL;
    }
    break;
  case kDLFloat:
    switch(dt.bits()) {
    case 32:
      return poplar::FLOAT;
    case 16:
      return poplar::HALF;
    }
    break;
  default:
    break;
  }
  LOG(FATAL) << "Unsupported data type for poplar:" << dt << "\n";
}


class PoplarCodeGen : public ExprVisitor {
public:
  explicit PoplarCodeGen() {}

  std::vector<poplar::program::Program> run(poplar::Graph& g, const ObjectRef& ref) {
    std::vector<poplar::program::Program> progs(3);
    curg_ = &g;
    progs[0] = poplar::program::Sequence();
    progs[1] = poplar::program::Sequence();
    progs_ = &progs;

    if (ref->IsInstance<FunctionNode>()) {
      LOG(WARNING) << "RUN FunctionNode";
      Function f = Downcast<Function>(ref);
      auto fname = GetExtSymbol(f);
      prog_map_[f] = 2;
      poplar::program::Sequence seq;
      progs[2] = seq;
      setup_args(f, 2, fname);

      curprog_ = &progs[2];
      curr_index_ = 2;
      this->VisitExpr(f);
      curr_index_ = -1;
      curprog_ = nullptr;

    } else if (ref->IsInstance<IRModuleNode>()) {
      LOG(WARNING) << "RUN IRModuleNode";
      IRModule mod = Downcast<IRModule>(ref);

      Function main = Downcast<Function>(mod->Lookup("main"));

      prog_map_[main] = 2;
      progs[2] = poplar::program::Sequence();
      setup_args(main, 2, "main");

      // First map the functions to progams
      for (const auto& it : mod->functions) {
        // We skip the "main" function since it was handled above
        LOG(WARNING) << "function " << it.first->name_hint;

        if (it.first->name_hint.compare("main") == 0)
          continue;

        Function f = Downcast<Function>(it.second);
        size_t index = progs.size();
        progs.push_back(poplar::program::Sequence());
        prog_map_[f] = index;
        setup_args(f, index, it.first->name_hint);
      }

      // Then convert the functions
      for (const auto& it : mod->functions) {
        // We skip the "main" function since it was handled above

        if (it.first->name_hint.compare("main") == 0)
          continue;

        Function f = Downcast<Function>(it.second);

        curprog_ = &progs[prog_map_[f]];
        this->VisitExpr(f);
        curprog_ = nullptr;
      }
    }

    // Make the programs to copy inputs/outputs

    prog_map_.clear();
    progs_ = nullptr;
    curg_ = nullptr;
    return progs;
  }

  /*!
   * \brief Get the external symbol of the Relay function name.
   *
   * \param func The provided function.
   *
   * \return An external symbol.
   */
  std::string GetExtSymbol(const Function& func) const {
    const auto name_node = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    CHECK(name_node.defined()) << "Fail to retrieve external symbol.";
    return std::string(name_node.value());
  }

  void VisitExpr_(const VarNode* node) {
  }
  void VisitExpr_(const GlobalVarNode* node) {
  }
  void VisitExpr_(const ConstantNode* node) {
  }
  void VisitExpr_(const TupleNode* node) {
  }
  void VisitExpr_(const FunctionNode* node) {
    LOG(WARNING) << "VISIT FunctionNode";
    this->VisitExpr(node->body);
  }
  void VisitExpr_(const CallNode* call) {
    LOG(WARNING) << "VISIT CallNode";
    if (IsOp(call, "add")) {
      CHECK_EQ(call->args.size(), 2);
      CHECK_GT(curr_index_, -1);
      auto& args = arg_map_[curr_index_ - 2];
      CHECK_EQ(args.size(), 3);
      auto& lhs = args[0];
      auto& rhs = args[1];
      auto& ret = args[2];

      auto seq_ = (poplar::program::Sequence*)curprog_;

      if (curr_index_ - 2 == 0) {
        // We are in the main function. We must connect input streams to function arguments.
        auto& streams = stream_map_[curr_index_ - 2];
        seq_->add(poplar::program::Copy(streams[0], lhs));
        seq_->add(poplar::program::Copy(streams[1], rhs));
        seq_->add(poplar::program::PrintTensor("lhs", lhs));
        seq_->add(poplar::program::PrintTensor("rhs", rhs));
        // And we must connect function return value to an output read tensor.
        curg_->createHostRead("fn_output_read", ret);
      }

      auto res = popops::add(*curg_, lhs, rhs, *seq_, "Add");
      seq_->add(poplar::program::Copy(res, ret));
      seq_->add(poplar::program::PrintTensor("ret", ret));
      LOG(WARNING) << "VISIT op +";
    } else if (IsOp(call, "subtract")) {
      LOG(WARNING) << "VISIT op -";
    } else if (IsOp(call, "multiply")) {
      LOG(WARNING) << "VISIT op *";
    } else {
      LOG(FATAL) << "Unrecognized op";
    }
  }
  void VisitExpr_(const LetNode* node) {
    LOG(WARNING) << "VISIT LetNode";
  }
  void VisitExpr_(const IfNode* node) {
    LOG(WARNING) << "VISIT IfNode";
  }
  void VisitExpr_(const OpNode* node) {
    LOG(WARNING) << "VISIT OpNode";
  }
  void VisitExpr_(const TupleGetItemNode* node) {
    LOG(WARNING) << "VISIT TupleGetItemNode";
  }
  void VisitExpr_(const RefCreateNode* node) {
    LOG(WARNING) << "VISIT RefCreateNode";
  }
  void VisitExpr_(const RefReadNode* node) {
    LOG(WARNING) << "VISIT RefReadNode";
  }
  void VisitExpr_(const RefWriteNode* node) {
    LOG(WARNING) << "VISIT RefWriteNode";
  }
  void VisitExpr_(const ConstructorNode* node) {
    LOG(WARNING) << "VISIT ConstructorNode";
  }
  void VisitExpr_(const MatchNode* node) {
    LOG(WARNING) << "VISIT MatchNode";
  }

private:
  void setup_args(Function fn, size_t index, const std::string& name) {
    LOG(WARNING) << "setup_args " << name;
    if (arg_map_.size() < (index - 1))
      arg_map_.resize(index - 1);
    auto& args = arg_map_[index-2];

    if (stream_map_.size() < (index - 1))
      stream_map_.resize(index - 1);
    auto& streams = stream_map_[index - 2];

    PoplarFunctionInfo pfi;
    pfi.program_index = index;
    for (const auto& it : fn->params) {
      std::string argName(it->vid->name_hint);
      LOG(WARNING) << argName;
      auto v = curg_->addVariable(poplar::FLOAT, {}, poplar::VariableMappingMethod::LINEAR, argName.c_str());
      auto stream = curg_->addHostToDeviceFIFO(argName + "-input-stream", poplar::FLOAT, 1);
      curg_->setTileMapping(v, 0);
      args.push_back(v);
      streams.push_back(stream);
	  pfi.arg_types.push_back(DLDataType{kDLFloat, 32, 1});
	  pfi.input_channels.push_back(argName);
    }
    // Last is the output
    auto v = curg_->addVariable(poplar::FLOAT, {}, poplar::VariableMappingMethod::LINEAR, "fn_output");
    curg_->setTileMapping(v, 0);
    args.push_back(v);
    pfi.arg_types.push_back(DLDataType{kDLFloat, 32, 1});
    pfi.output_channel = "fn_output";
    poplar_function_info[name] = pfi;
  }

  poplar::Graph* curg_;
  poplar::program::Program* curprog_;
  int curr_index_;
  std::vector<poplar::program::Program>* progs_;
  std::map<Function, size_t> prog_map_;
  std::vector<std::vector<poplar::Tensor>> arg_map_;
  std::vector<std::vector<poplar::DataStream>> stream_map_;
public:
  std::unordered_map<std::string, PoplarFunctionInfo> poplar_function_info;
};

runtime::Module PoplarCompiler(const ObjectRef& ref) {
  // XXX: We need some way for the user to configure this
  // poplar::Target t = poplar::Target::createIPUTarget(1, "ipu1");
  poplar::Target t = tvm::runtime::contrib::IPUDeviceAPI::Global()->getTarget();
  poplar::Graph g(t);
  popops::addCodelets(g);
  PoplarCodeGen codegen;
  auto progs = codegen.run(g, ref);
  // XXX: This needs to be filled in
  std::unordered_map<std::string, PoplarFunctionInfo>& fn_map = codegen.poplar_function_info;

  // Compile
  poplar::Executable exe = poplar::compileGraph(g, progs);

  const auto* pf = runtime::Registry::Get("module.poplar_module_create");
  CHECK(pf != nullptr) << "Cannot find Poplar module to create the external runtime module";
  return (*pf)(static_cast<void*>(&exe), static_cast<void*>(&fn_map));
}

TVM_REGISTER_GLOBAL("relay.ext.poplar").set_body_typed(PoplarCompiler);

}
}
}
