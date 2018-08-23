import numpy as np
import tvm
import topi
import topi.testing
from topi import util


def test_logic_ewise():
    m = tvm.var('m')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), dtype='uint8', name='A')

    shape = (20, 3)

    def test_apply(func, name, f_numpy):
        B = func(A)
        assert tuple(B.shape) == tuple(A.shape)
        a_np = np.random.randint(low=0, high=2, size=shape, dtype='uint8')
        b_np = f_numpy(a_np).astype('uint8')

        def check_device(device):
            ctx = tvm.context(device, 0)
            if not ctx.exist:
                print("Skip because %s is not enabled" % device)
                return
            print("Running on target: %s" % device)
            with tvm.target.create(device):
                s = topi.generic.schedule_injective(B)
            foo = tvm.build(s, [A, B], device, name=name)
            a = tvm.nd.array(a_np, ctx)
            b = tvm.nd.array(np.zeros_like(b_np), ctx)
            foo(a, b)
            np.testing.assert_equal(b.asnumpy(), b_np)

        for device in ['cuda', 'opencl', 'metal', 'rocm', 'vulkan', 'llvm', 'nvptx', 'sdaccel']:
            check_device(device)


    test_apply(topi.logical_not, "not", np.logical_not)

if __name__ == "__main__":
    test_logic_ewise()
