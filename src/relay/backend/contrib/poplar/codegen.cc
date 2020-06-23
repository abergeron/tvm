#include <poplar/Graph.hpp>
#include <poplar/Target.hpp>
#include <poplar/Program.hpp>

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>

namespace tvm {
namespace relay {
namespace contrib {

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

      BaseFunc main = mod->Lookup("main");
      prog_map_[main] = 2;
      progs[2] = poplar::program::Sequence();

      // First map the functions to progams
      for (const auto& it : mod->functions) {
	// We skip the "main" function since it was handled above
	if (it.first->name_hint.compare("main") == 0)
	  continue;

	size_t index = progs.size();
	progs.push_back(poplar::program::Sequence());
	prog_map_[it.second] = index;
      }

      // Main is special since its input/output go to the outside world


      // Then convert the functions
      for (const auto& it : mod->functions) {
	// We skip the "main" function since it was handled above
	if (it.first->name_hint.compare("main") == 0)
	  continue;

	curprog_ = &progs[prog_map_[it.second]];
	this->VisitExpr(Downcast<Function>(it.second));
	curprog_ = nullptr;
      }
    }

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
  poplar::Graph* curg_;
  poplar::program::Program* curprog_;
  std::vector<poplar::program::Program>* progs_;
  std::map<BaseFunc, size_t> prog_map_;
};

runtime::Module PoplarCompiler(const ObjectRef& ref) {
  poplar::Target t = poplar::Target::createIPUTarget(1, "C2");
  poplar::Graph g(t);
  PoplarCodeGen codegen;
  auto progs = codegen.run(g, ref);
  const auto* pf = runtime::Registry::Get("module.poplar_module_create");
  CHECK(pf != nullptr) << "Cannot fine Poplar module to create the external runtime module";
  return (*pf)(static_cast<void*>(&g), static_cast<void*>(&progs));
}

TVM_REGISTER_GLOBAL("relay.ext.poplar").set_body_typed(PoplarCompiler);

}
}
}
