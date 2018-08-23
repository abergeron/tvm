"""Logical operators"""
from __future__ import absolute_import as _abs
from .import cpp as _cpp


def logical_not(x):
    """Take the logical negation of x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return _cpp.logical_not(x)
