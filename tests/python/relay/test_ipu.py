"""IPU tests.
Copied here to allow to set IPU as target.
"""

import os
import sys
import tvm
from tvm import relay
from tvm.relay import testing
from tvm.relay.op import add
from tvm.contrib import util
from tvm import runtime
import numpy as np

def check_result(mod, map_inputs, out_shape, result, tol=1e-5, target="llvm",
                 ctx=tvm.cpu()):
    with tvm.transform.PassContext(opt_level=3,
                                   disabled_pass=["AlterOpLayout"]):
        vm = relay.create_executor(kind="vm", mod=mod, ctx=ctx, target=target)
    out = vm.evaluate()(**map_inputs)
    tvm.testing.assert_allclose(out.asnumpy(), result, rtol=tol, atol=tol)


def wrap_fn(func, name):
    func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Compiler", "poplar")
    func = func.with_attr("global_symbol", name)
    new_vars = [relay.var(a.name_hint, a.type_annotation) for a in func.params]
    return relay.Call(func, new_vars)


def test_add_op_scalar():
    """
    Program:
        fn (x, y) {
            return x + y;
        }
    """
    dtype = 'float32'
    x = relay.var('x', shape=(4,), dtype=dtype)
    y = relay.var('y', shape=(4,), dtype=dtype)
    z = add(x, y)
    func = relay.Function([x, y], z)
    wrap = wrap_fn(func, "pop_add")
    mod = tvm.IRModule.from_expr(wrap)
    x_data = np.array([10, -1, 3.5, -2  ], dtype=dtype)
    y_data = np.array([ 1,  1, 4.7, 17.4], dtype=dtype)
    check_result(mod, {"x": x_data, "y": y_data}, (4,), x_data + y_data)


def test_mlp():
    image_shape = (1, 1, 28, 28)
    func = testing.mlp.get_net(1)
    wrap = wrap_fn(func, "pop_mlp")
    mod, params = testing.create_workload(wrap)
    check_result(mod, params, (1, 10), 0)

    #benchmark_execution(mod, params, data_shape=image_shape, out_shape=(1, 10), model="mlp")
