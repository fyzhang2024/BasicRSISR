import sys
import types

import torch.nn as nn


def ensure_minimal_thop_compat():
    try:
        import thop  # noqa: F401
        return
    except ImportError:
        pass

    thop_mod = sys.modules.setdefault("thop", types.ModuleType("thop"))

    def profile(*args, **kwargs):
        return 0, 0

    thop_mod.profile = profile


def ensure_minimal_basicsr_compat():
    ensure_minimal_thop_compat()
    try:
        import basicsr  # noqa: F401
        return
    except ImportError:
        pass

    class RegistryStub(object):
        def register(self, *args, **kwargs):
            def decorator(cls):
                return cls
            return decorator

    def to_2tuple(x):
        if isinstance(x, tuple):
            return x
        return (x, x)

    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

    basicsr_mod = sys.modules.setdefault("basicsr", types.ModuleType("basicsr"))
    utils_mod = sys.modules.setdefault("basicsr.utils", types.ModuleType("basicsr.utils"))
    registry_mod = sys.modules.setdefault("basicsr.utils.registry", types.ModuleType("basicsr.utils.registry"))
    archs_mod = sys.modules.setdefault("basicsr.archs", types.ModuleType("basicsr.archs"))
    arch_util_mod = sys.modules.setdefault("basicsr.archs.arch_util", types.ModuleType("basicsr.archs.arch_util"))

    basicsr_mod.utils = utils_mod
    basicsr_mod.archs = archs_mod
    utils_mod.registry = registry_mod
    archs_mod.arch_util = arch_util_mod

    registry_mod.ARCH_REGISTRY = RegistryStub()
    arch_util_mod.to_2tuple = to_2tuple
    arch_util_mod.trunc_normal_ = trunc_normal_
