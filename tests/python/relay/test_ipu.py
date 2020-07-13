# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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


def get_result(fn, params):
    mod = tvm.IRModule.from_expr(fn)
    e = relay.create_executor(kind='debug', mod=mod, ctx=tvm.cpu(), target="llvm")
    return e.evaluate()(**params).asnumpy()


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
    input_data = np.random.uniform(size=image_shape).astype('float32')
    params["data"] = input_data

    # We use the original func here, not the poplar-tagged one
    result = get_result(func, params)

    check_result(mod, params, (1, 10), result)


def test_if():
    i = relay.var('i', shape=[], dtype='int32')
    sb = relay.ScopeBuilder()
    with sb.if_scope(relay.greater_equal(i, relay.const(0, dtype='int32'))):
        sb.ret(i)
    with sb.else_scope():
        one_more = relay.add(i, relay.const(1, dtype='int32'))
        sb.ret(one_more)

    func = relay.Function([i], sb.get())
    wrap = wrap_fn(func, "pop_if")
    mod = tvm.IRModule.from_expr(wrap)

    check_result(mod, {'i': -2}, (), -1)
    check_result(mod, {'i': 0}, (), 0)
    check_result(mod, {'i': 2}, (), 2)
