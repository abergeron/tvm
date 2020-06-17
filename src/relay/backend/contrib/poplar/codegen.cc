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
  explicit PoplarCodeGen();

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
};

}
}
}
