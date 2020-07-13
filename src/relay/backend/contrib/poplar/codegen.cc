/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Program.hpp>
#include <poplar/Target.hpp>
#include <poplar/Type.hpp>
#include <poplar/VariableMappingMethod.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popnn/NonLinearity.hpp>
#include <popnn/NonLinearityDef.hpp>
#include <popnn/codelets.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

#include "../../../../runtime/contrib/poplar/fn_info.h"
#include "../../utils.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

using tvm::runtime::contrib::pop_fn_info;
using tvm::runtime::contrib::PoplarFunctionInfo;

static poplar::Type to_poplar_dtype(const Type& t) {
  const TensorTypeNode* tt = t.as<TensorTypeNode>();
  CHECK(tt != nullptr) << "Only support tensor types for now.\n";
  DataType dt = tt->dtype;
  if (dt.lanes() != 1) LOG(FATAL) << "Poplar doesn't support multi-lane data types (for now)\n";
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
      switch (dt.bits()) {
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
  PoplarCodeGen() {}

  std::pair<std::vector<poplar::program::Program>, pop_fn_info> run(poplar::Graph* g,
                                                                    const ObjectRef& ref) {
    curg_ = g;

    // We need to add codelets (aka ops) for the libs that we use to the Graph
    poplin::addCodelets(*curg_);
    popnn::addCodelets(*curg_);
    popops::addCodelets(*curg_);

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

      // The first loop maps the function to polar Program, and the
      // second loops converts the functions.  This is to ensure that
      // if a function calls the other, the program is there to be
      // called (although there is no support for function calls for now).

      for (const auto& it : mod->functions) {
        Function f = Downcast<Function>(it.second);
        this->setup_fn(f, it.first->name_hint);
      }

      for (const auto& it : mod->functions) {
        Function f = Downcast<Function>(it.second);
        this->VisitExpr(f);

        const auto name_node = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
        if (name_node.defined()) this->fill_fn_info(f, name_node.value());
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

  void VisitExpr_(const GlobalVarNode* node) { CHECK(false) << "GlobalVarNode not supported\n"; }

  void VisitExpr_(const ConstantNode* node) {
    poplar::Type t = to_poplar_dtype(node->checked_type());
    poplar::Tensor res;
    // poplar need the type of the input as a template parameter so we
    // can't just feed void*
    if (t == poplar::CHAR || t == poplar::SIGNED_CHAR) {
      res = curg_->addConstant(t, to_poplar_shape(node->checked_type()),
                               static_cast<int8_t*>(node->data->data));
    } else if (t == poplar::UNSIGNED_CHAR) {
      res = curg_->addConstant(t, to_poplar_shape(node->checked_type()),
                               static_cast<uint8_t*>(node->data->data));
    } else if (t == poplar::SHORT) {
      res = curg_->addConstant(t, to_poplar_shape(node->checked_type()),
                               static_cast<int16_t*>(node->data->data));
    } else if (t == poplar::UNSIGNED_SHORT) {
      res = curg_->addConstant(t, to_poplar_shape(node->checked_type()),
                               static_cast<uint16_t*>(node->data->data));
    } else if (t == poplar::INT) {
      res = curg_->addConstant(t, to_poplar_shape(node->checked_type()),
                               static_cast<int32_t*>(node->data->data));
    } else if (t == poplar::UNSIGNED_INT) {
      res = curg_->addConstant(t, to_poplar_shape(node->checked_type()),
                               static_cast<uint32_t*>(node->data->data));
    } else if (t == poplar::HALF) {
      res = curg_->addConstant(t, to_poplar_shape(node->checked_type()),
                               static_cast<uint16_t*>(node->data->data));
    } else if (t == poplar::FLOAT) {
      res = curg_->addConstant(t, to_poplar_shape(node->checked_type()),
                               static_cast<float*>(node->data->data));
    } else {
      CHECK(false) << "Unhandled type in ConstantNode\n";
    }
    poputil::mapTensorLinearly(*curg_, res);
    expr_map_[node] = res;
  }

  void VisitExpr_(const TupleNode* node) { CHECK(false) << "TupleNode not supported\n"; }

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
      CHECK(false) << "Calling a function not supported for now";
    } else {
      // it's an op
      if (IsOp(call, "add")) {
        CHECK_EQ(call->args.size(), 2);
        expr_map_[call] = popops::add(*curg_, expr_map_[call->args[0].get()],
                                      expr_map_[call->args[1].get()], *curp_);
      } else if (IsOp(call, "nn.batch_flatten")) {
        CHECK_EQ(call->args.size(), 1);
        const poplar::Tensor& arg = expr_map_[call->args[0].get()];
        expr_map_[call] = arg.flatten(1, arg.rank());
      } else if (IsOp(call, "nn.dense")) {
        CHECK_EQ(call->args.size(), 2);
        expr_map_[call] = poplin::matMul(*curg_, expr_map_[call->args[0].get()],
                                         expr_map_[call->args[1].get()].transpose(), *curp_);
      } else if (IsOp(call, "nn.relu")) {
        CHECK_EQ(call->args.size(), 1);
        expr_map_[call] = popnn::nonLinearity(*curg_, popnn::NonLinearityType::RELU,
                                              expr_map_[call->args[0].get()], *curp_);
      } else if (IsOp(call, "nn.softmax")) {
        CHECK_EQ(call->args.size(), 1);
        expr_map_[call] = popnn::nonLinearity(*curg_, popnn::NonLinearityType::SOFTMAX,
                                              expr_map_[call->args[0].get()], *curp_);
      } else if (IsOp(call, "greater_equal")) {
        CHECK_EQ(call->args.size(), 2);
        expr_map_[call] = popops::gteq(*curg_, expr_map_[call->args[0].get()],
                                       expr_map_[call->args[1].get()], *curp_);
      } else {
        LOG(FATAL) << "Unrecognized op: " << PrettyPrint(call->op);
      }
    }
  }

  void VisitExpr_(const LetNode* node) { LOG(WARNING) << "VISIT LetNode"; }

  void VisitExpr_(const IfNode* node) {
    auto* bak = curp_;
    poplar::program::Sequence true_body;
    poplar::program::Sequence false_body;
    // This tensor is used to "merge" the resuts of the two branches
    auto result = curg_->addVariable(to_poplar_dtype(node->checked_type()),
                                     to_poplar_shape(node->checked_type()),
                                     poplar::VariableMappingMethod::LINEAR, "");

    // The way this is done might cause problems if the two branches
    // share some code that has not been visited previously, since it
    // will be mapped to one of the branches and not execute on the
    // other.  The tensor will still be there, but they will be
    // garbage-initialized and this will produce bad results. Not sure
    // how to fix that without extensive analysis.

    // Make sure the condition is mapped
    this->VisitExpr(node->cond);

    // Set the current program to a new one and visit the true branch
    // to build the true program, then copy the results to the
    // "output" tensor
    curp_ = &true_body;
    this->VisitExpr(node->true_branch);
    curp_->add(poplar::program::Copy(expr_map_[node->true_branch.get()], result));

    // Set the current program to a new one and visit the false branch
    // to build the false program, then copy the results to the
    // "output" tensor
    curp_ = &false_body;
    this->VisitExpr(node->false_branch);
    curp_->add(poplar::program::Copy(expr_map_[node->false_branch.get()], result));

    // Set the current program back to the original one and insert the If node.
    curp_ = bak;
    curp_->add(poplar::program::If(expr_map_[node->cond.get()], true_body, false_body));
    expr_map_[node] = result;
  }
  void VisitExpr_(const OpNode* node) { LOG(WARNING) << "VISIT OpNode"; }

  void VisitExpr_(const TupleGetItemNode* node) { LOG(WARNING) << "VISIT TupleGetItemNode"; }

  void VisitExpr_(const RefCreateNode* node) { LOG(WARNING) << "VISIT RefCreateNode"; }

  void VisitExpr_(const RefReadNode* node) { LOG(WARNING) << "VISIT RefReadNode"; }

  void VisitExpr_(const RefWriteNode* node) { LOG(WARNING) << "VISIT RefWriteNode"; }

  void VisitExpr_(const ConstructorNode* node) { LOG(WARNING) << "VISIT ConstructorNode"; }

  void VisitExpr_(const MatchNode* node) { LOG(WARNING) << "VISIT MatchNode"; }

 private:
  void setup_fn(Function fn, const std::string& name) {
    // Adds the mapping for the arguments of a function
    size_t index = progs_.size();
    progs_.push_back(poplar::program::Sequence());
    prog_map_[fn.get()] = index;
    for (const auto& it : fn->params) {
      expr_map_[it.get()] = curg_->addVariable(
          to_poplar_dtype(it->checked_type()), to_poplar_shape(it->checked_type()),
          poplar::VariableMappingMethod::LINEAR, it->vid->name_hint.c_str());
    }
  }

  void fill_fn_info(const Function& f, const std::string& name) {
    // Fills in a PoplarFunctionInfor for a function and setup its
    // input and output tensors to be reachable from the outside.
    // This has to be done after code generation is complete.

    // The way this is done is probably sub-optimal.  The best way
    // would be to generate a program that does the copies from
    // streams for the arguments, call the specified function and the
    // copies the result out.  This would also require code changes in
    // the module to use the streams instead of {write,read}Tensor.

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

  // This is not the best interface, but there is currrently no way to
  // pass information to a backend.
  char* tmp = getenv("TVM_POPLAR_NUM_IPU");
  if (tmp != NULL) num_ipu = std::atoi(tmp);

  tmp = getenv("TVM_POPLAR_USE_MODEL");
  if (tmp != NULL) use_model = static_cast<bool>(std::atoi(tmp));

  poplar::Target t;
  if (use_model) {
    poplar::IPUModel m;
    t = m.createDevice().getTarget();
  } else {
    t = poplar::Target::createIPUTarget(num_ipu, "ipu1");
  }
  // We create the Graph here to avoid shenanigans with std::move()
  // since it is not copyable.
  poplar::Graph g(t);
  PoplarCodeGen codegen;
  auto result = codegen.run(&g, ref);
  const auto& progs = result.first;
  pop_fn_info& fn_map = result.second;

  // Compile
  poplar::Executable exe = poplar::compileGraph(g, progs);

  const auto* pf = runtime::Registry::Get("module.poplar_module_create");
  CHECK(pf != nullptr) << "Cannot find Poplar module to create the external runtime module";
  return (*pf)(static_cast<void*>(&exe), static_cast<void*>(&fn_map));
}

TVM_REGISTER_GLOBAL("relay.ext.poplar").set_body_typed(PoplarCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
