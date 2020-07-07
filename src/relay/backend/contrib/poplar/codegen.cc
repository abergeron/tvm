#include <poplar/Graph.hpp>
#include <poplar/Target.hpp>
#include <poplar/Program.hpp>
#include <poplar/Type.hpp>
#include <poplar/VariableMappingMethod.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>

#include <popnn/codelets.hpp>
#include <popnn/NonLinearityDef.hpp>
#include <popnn/NonLinearity.hpp>

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>
#include "../../utils.h"

#include "../../../../runtime/contrib/poplar/fn_info.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

using tvm::runtime::contrib::PoplarFunctionInfo;
using tvm::runtime::contrib::pop_fn_info;

static poplar::Type to_poplar_dtype(const Type& t) {
  const TensorTypeNode* tt = t.as<TensorTypeNode>();
  CHECK(tt != nullptr) << "Only support tensor types for now.\n";
  DataType dt = tt->dtype;
  if (dt.lanes() != 1)
    LOG(FATAL) << "Poplar doesn't support multi-lane data types (for now)\n";
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

static std::vector<size_t> to_poplar_shape(const Type& t) {
  const TensorTypeNode* tt = t.as<TensorTypeNode>();
  CHECK(tt != nullptr) << "Only support tensor types for now.\n";
  std::vector<size_t> res;
  for (const auto& it : tt->shape) {
    IntImm i = Downcast<IntImm>(it);
    res.push_back(i->value);
  }
  return res;
}

class PoplarCodeGen : public ExprVisitor {
public:
  explicit PoplarCodeGen() {}

  std::pair<std::vector<poplar::program::Program>, pop_fn_info> run(poplar::Graph& g, const ObjectRef& ref) {
    curg_ = &g;

    popops::addCodelets(g);
    popnn::addCodelets(g);

    if (ref->IsInstance<FunctionNode>()) {
      Function f = Downcast<Function>(ref);
      this->setup_fn(f, "");
      this->VisitExpr(f);

      const auto name_node = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
      CHECK(name_node.defined()) << "Function doesn't have global symbol";
      this->fill_fn_info(f, name_node.value());

    } else if (ref->IsInstance<IRModuleNode>()) {
      // This seems never executed for now, thus may not be up-to-date.
      CHECK(false) << "Hit IRModule case\n";

      IRModule mod = Downcast<IRModule>(ref);

      // First map the functions to progams
      for (const auto& it : mod->functions) {
        Function f = Downcast<Function>(it.second);
        this->setup_fn(f, it.first->name_hint);
      }

      // Then convert the functions
      for (const auto& it : mod->functions) {
        Function f = Downcast<Function>(it.second);
        this->VisitExpr(f);

	const auto name_node = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
	if (name_node.defined())
	  this->fill_fn_info(f, name_node.value());
      }
    }

    // Clear temp state
    prog_map_.clear();
    expr_map_.clear();
    curg_ = nullptr;
    curp_ = nullptr;

    // This clears and returns the returned state
    return std::make_pair(std::move(progs_), std::move(fn_info_));
  }

  void VisitExpr_(const VarNode* node) {
    // Safety check to make sure everything is ok.
    CHECK(expr_map_.find(node) != expr_map_.end());
  }
  void VisitExpr_(const GlobalVarNode* node) {
    CHECK(false) << "Seen a GlobalVarNode\n";
  }
  void VisitExpr_(const ConstantNode* node) {
    CHECK(false) << "Seen a ConstantNode\n";
  }
  void VisitExpr_(const TupleNode* node) {
    CHECK(false) << "Seen a TupleNode\n";
  }
  void VisitExpr_(const FunctionNode* node) {
    curp_ = static_cast<poplar::program::Sequence*>(&progs_[prog_map_[node]]);
    this->VisitExpr(node->body);
  }
  void VisitExpr_(const CallNode* call) {
    // We need to visit expressions first to populate the map
    for (const auto& a : call->args) {
      this->VisitExpr(a);
    }
    if (const auto* func = call->op.as<FunctionNode>()) {
      // function call
      CHECK(false) << "not supported for now";
    } else {
      // it's an op
      auto* curseq = static_cast<poplar::program::Sequence*>(curp_);
      if (IsOp(call, "add")) {
	CHECK_EQ(call->args.size(), 2);
	expr_map_[call] =
	  popops::add(*curg_, expr_map_[call->args[0].get()], expr_map_[call->args[1].get()], *curseq);
      } else if (IsOp(call, "subtract")) {
	LOG(WARNING) << "VISIT op -";
      } else if (IsOp(call, "multiply")) {
	LOG(WARNING) << "VISIT op *";
      } else if (IsOp(call, "nn.softmax")) {
	CHECK_EQ(call->args.size(), 1);
	expr_map_[call] =
	  popnn::nonLinearity(*curg_, popnn::NonLinearityType::SOFTMAX, expr_map_[call->args[0].get()], *curseq);
      } else {
	LOG(FATAL) << "Unrecognized op: " << PrettyPrint(call->op);
      }
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
  void setup_fn(Function fn, const std::string& name) {
    size_t index = progs_.size();
    progs_.push_back(poplar::program::Sequence());
    prog_map_[fn.get()] = index;
    for (const auto& it : fn->params) {
      expr_map_[it.get()] = curg_->addVariable(to_poplar_dtype(it->checked_type()),
					       to_poplar_shape(it->checked_type()),
					       poplar::VariableMappingMethod::LINEAR,
					       it->vid->name_hint.c_str());
    }
  }

  void fill_fn_info(const Function& f, const std::string& name) {
    PoplarFunctionInfo& pfi = fn_info_[name];
    pfi.program_index = prog_map_[f.get()];
    size_t i = 0;
    for (const auto& it : f->params) {
      std::string sname = name + "_input" + std::to_string(i);
      curg_->createHostWrite(sname, expr_map_[it.get()]);
      pfi.input_channels.push_back(sname);
      i++;
    }
    pfi.output_channel = name + "_output";
    curg_->createHostRead(pfi.output_channel, expr_map_[f->body.get()]);
  }

  // temp state
  poplar::Graph* curg_;
  poplar::program::Sequence* curp_;

  // Conversion state
  std::unordered_map<const FunctionNode*, size_t> prog_map_;
  std::unordered_map<const ExprNode*, poplar::Tensor> expr_map_;

  // returned state
  std::vector<poplar::program::Program> progs_;
  pop_fn_info fn_info_;
};

runtime::Module PoplarCompiler(const ObjectRef& ref) {
  int num_ipu = 1;
  bool use_model = false;
  char* tmp = getenv("TVM_POPLAR_NUM_IPU");
  if (tmp != NULL)
    num_ipu = std::atoi(tmp);

  tmp = getenv("TVM_POPLAR_USE_MODEL");
  if (tmp != NULL)
    use_model = bool(std::atoi(tmp));

  poplar::Target t;
  if (use_model) {
    poplar::IPUModel m;
    t = m.createDevice().getTarget();
  } else {
    t = poplar::Target::createIPUTarget(num_ipu, "ipu1");
  }
  poplar::Graph g(t);
  PoplarCodeGen codegen;
  auto result = codegen.run(g, ref);
  const auto& progs = result.first;
  pop_fn_info& fn_map = result.second;

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
