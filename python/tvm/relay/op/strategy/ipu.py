import topi
from .generic import *


@schedule_injective.register("ipu")
def schedule_injective_cpu(attrs, outs, target):
    """schedule injective ops for ipu"""
    with target:
        return topi.ipu.schedule_injective(outs)
