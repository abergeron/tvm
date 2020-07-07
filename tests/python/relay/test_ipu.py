"""IPU tests.
Copied here to allow to set IPU as target.
"""

import os
import sys
import tvm
from tvm import relay
from tvm.relay.op import add
from tvm.contrib import util
from tvm import runtime
import numpy as np

def check_result(mod, map_inputs, out_shape, result, tol=1e-5, target="llvm",
                 ctx=tvm.cpu()):

    def check_vm_result():
        with tvm.transform.PassContext(opt_level=3,
                                       disabled_pass=["AlterOpLayout"]):
            vm = relay.create_executor(kind="vm", mod=mod, ctx=ctx, target=target)
        out = vm.evaluate()(**map_inputs)
        tvm.testing.assert_allclose(out.asnumpy(), result, rtol=tol, atol=tol)

    check_vm_result()


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
    x0 = relay.var('x0', shape=(4,), dtype=dtype)
    y0 = relay.var('y0', shape=(4,), dtype=dtype)
    z0 = add(x0, y0)
    func = relay.Function([x0, y0], z0)
    func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Compiler", "poplar")
    func = func.with_attr("global_symbol", "pop_add")
    call = relay.Call(func, [x, y])
    mod = tvm.IRModule.from_expr(call)
    x_data = np.array([10, -1, 3.5, -2  ], dtype=dtype)
    y_data = np.array([ 1,  1, 4.7, 17.4], dtype=dtype)
    check_result(mod, {"x": x_data, "y": y_data}, (), x_data + y_data, target='llvm')


