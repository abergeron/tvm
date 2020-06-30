#include <poplar/Graph.hpp>
#include <poplar/Target.hpp>
#include <poplar/Program.hpp>
#include <poplar/Type.hpp>
#include <poplar/VariableMappingMethod.hpp>
#include <poplar/Engine.hpp>

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>

#include "../../../../runtime/contrib/poplar/fn_info.h"

namespace tvm {
namespace relay {
namespace contrib {

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
      progs[2] = poplar::program::Sequence();
      curprog_ = &progs[2];
      this->VisitExpr(Downcast<Function>(ref));
      curprog_ = nullptr;

    } else if (ref->IsInstance<IRModuleNode>()) {
      IRModule mod = Downcast<IRModule>(ref);

      Function main = Downcast<Function>(mod->Lookup("main"));

      prog_map_[main] = 2;
      progs[2] = poplar::program::Sequence();
      setup_args(main, 2);

      // First map the functions to progams
      for (const auto& it : mod->functions) {
        // We skip the "main" function since it was handled above

        if (it.first->name_hint.compare("main") == 0)
          continue;

        Function f = Downcast<Function>(it.second);
        size_t index = progs.size();
        progs.push_back(poplar::program::Sequence());
        prog_map_[f] = index;
        setup_args(f, index);
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

  void VisitExpr_(const VarNode* node) {}
  void VisitExpr_(const GlobalVarNode* node) {}
  void VisitExpr_(const ConstantNode* node) {}
  void VisitExpr_(const TupleNode* node) {}
  void VisitExpr_(const FunctionNode* node) {}
  void VisitExpr_(const CallNode* node) {}
  void VisitExpr_(const LetNode* node) {}
  void VisitExpr_(const IfNode* node) {}
  void VisitExpr_(const OpNode* node) {}
  void VisitExpr_(const TupleGetItemNode* node) {}
  void VisitExpr_(const RefCreateNode* node) {}
  void VisitExpr_(const RefReadNode* node) {}
  void VisitExpr_(const RefWriteNode* node) {}
  void VisitExpr_(const ConstructorNode* node) {}
  void VisitExpr_(const MatchNode* node) {}

private:
  void setup_args(Function fn, size_t index) {
    if (arg_map_.size() < (index - 1))
      arg_map_.resize(index - 1);
    auto& args = arg_map_[index-2];
    for (const auto& it : fn->params) {
      args.push_back(curg_->addVariable(poplar::FLOAT, {}, poplar::VariableMappingMethod::LINEAR,
					it->vid->name_hint.c_str()));
    }
    // Last is the output
    args.push_back(curg_->addVariable(poplar::FLOAT, {}, poplar::VariableMappingMethod::LINEAR,
				      "fn_output"));
  }

  poplar::Graph* curg_;
  poplar::program::Program* curprog_;
  std::vector<poplar::program::Program>* progs_;
  std::map<Function, size_t> prog_map_;
  std::vector<std::vector<poplar::Tensor>> arg_map_;
};

runtime::Module PoplarCompiler(const ObjectRef& ref) {
  // XXX: We need some way for the user to configure this
  poplar::Target t = poplar::Target::createIPUTarget(1, "ipu1");
  poplar::Graph g(t);
  PoplarCodeGen codegen;
  // XXX: This needs to be filled in
  std::map<std::string, PoplarFunctionInfo> fn_map;
  auto progs = codegen.run(g, ref);

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
