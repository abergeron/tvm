"""IPU tests.
Copied here to allow to set IPU as target.
"""

import sys
import tvm
from tvm import relay
from tvm.relay.op import add
import numpy as np

def check_result(mod, map_inputs, out_shape, result, tol=1e-5, target="llvm",
                 ctx=tvm.cpu()):
    if sys.platform == "win32":
        print("Skip test on Windows for now")
        return

    def update_lib(lib):
        test_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
        source_dir = os.path.join(test_dir, "..", "..", "..")
        contrib_path = os.path.join(source_dir, "src", "runtime", "contrib")

        kwargs = {}
        kwargs["options"] = ["-O2", "-std=c++14", "-I" + contrib_path]
        tmp_path = util.tempdir()
        lib_name = 'lib.so'
        lib_path = tmp_path.relpath(lib_name)
        lib.export_library(lib_path, fcompile=False, **kwargs)
        lib = tvm.runtime.load_module(lib_path)

        return lib

    def check_vm_result():
        with tvm.transform.PassContext(opt_level=3,
                                       disabled_pass=["AlterOpLayout"]):
            exe = relay.vm.compile(mod, target=target)
        code, lib = exe.save()
        lib = update_lib(lib)
        exe = runtime.vm.Executable.load_exec(code, lib)
        vm = runtime.vm.VirtualMachine(exe)
        vm.init(ctx)
        out = vm.run(**map_inputs)
        tvm.testing.assert_allclose(out.asnumpy(), result, rtol=tol, atol=tol)

    def check_graph_runtime_result():
        with tvm.transform.PassContext(opt_level=3,
                                       disabled_pass=["AlterOpLayout"]):
            json, lib, _ = relay.build(mod, target=target)
        lib = update_lib(lib)
        rt_mod = tvm.contrib.graph_runtime.create(json, lib, ctx)

        for name, data in map_inputs.items():
            rt_mod.set_input(name, data)
        rt_mod.run()
        out = tvm.nd.empty(out_shape, ctx=ctx)
        out = rt_mod.get_output(0, out)

        tvm.testing.assert_allclose(out.asnumpy(), result, rtol=tol, atol=tol)

    check_vm_result()
    check_graph_runtime_result()

def test_add_op_scalar():
    """
    Program:
        fn (x, y) {
            return x + y;
        }
    """
    x = relay.var('x', shape=())
    y = relay.var('y', shape=())
    x0 = relay.var('x0', shape=())
    y0 = relay.var('y0', shape=())
    z0 = add(x0, y0)
    func = relay.Function([x0, y0], z0)
    func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Compiler", "poplar")
    func = func.with_attr("global_symbol", "main")
    call = relay.Call(func, [x, y])
    mod = tvm.IRModule.from_expr(call)
    x_data = np.array(10.0, dtype='float32')
    y_data = np.array(1.0, dtype='float32')
    check_result(mod, {"x": x_data, "y": y_data}, (), x_data + y_data, target='cpu')
