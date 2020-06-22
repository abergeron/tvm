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
    std::vector<poplar::program::Program> progs;
    curg_ = &g;

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
  std::vector<poplar::program::Program> progs_;
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
