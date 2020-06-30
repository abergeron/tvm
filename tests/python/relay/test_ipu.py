"""IPU tests.
Copied here to allow to set IPU as target.
"""

import tvm
from tvm import relay
from tvm.relay.op import add
import numpy as np

def get_ipu_target_and_context():
    target = "ipu"
    ctx = tvm.context(target, 0)
    assert ctx.exist
    return target, ctx

# @tq, @jr should we put this in testing ns?
def check_rts(expr, args, expected_result, mod=None, target=None, ctx=None):
    """
    Check that evaluating `expr` applied to the arguments produces
    `result` on both the evaluator and TVM runtime.

    Parameters
    ----------
    expr:
        The expression to evaluate

    args: list of Expr
        The arguments to supply the expr.

    expected_result:
        The expected result of running the expression.
    """
    executor_kwargs = {}
    if target is not None:
        executor_kwargs['target'] = target
    if ctx is not None:
        executor_kwargs['ctx'] = ctx
    intrp = relay.create_executor('debug', mod=mod, **executor_kwargs)
    graph = relay.create_executor('graph', mod=mod, **executor_kwargs)
    eval_result = intrp.evaluate(expr)(*args)
    rts_result = graph.evaluate(expr)(*args)
    tvm.testing.assert_allclose(eval_result.asnumpy(), rts_result.asnumpy())
    tvm.testing.assert_allclose(eval_result.asnumpy(), expected_result)

def test_add_op_scalar():
    """
    Program:
        fn (x, y) {
            return x + y;
        }
    """
    x = relay.var('x', shape=())
    y = relay.var('y', shape=())
    func = relay.Function([x, y], add(x, y))
    x_data = np.array(10.0, dtype='float32')
    y_data = np.array(1.0, dtype='float32')

    target, ctx = get_ipu_target_and_context()
    check_rts(func, [x_data, y_data], x_data + y_data, target=target, ctx=ctx)
