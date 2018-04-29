from ..java_classes import *
from ..ndarray import array, ndarray


_SDVariable_class = 'org.nd4j.autodiff.samediff.SDVariable'


def _is_sdvar(x):
	return type(x).__name__ == _SDVariable_class


def _is_jumpy(x):
	return type(x).__name__ == 'SDVariableWrapper'


def op(f):
    def wrapper(*args, **kwargs):
        args = list(args)
        for i, a in enumerate(args):
            if _is_jumpy(a):
                args[i] = a.var
        for k in kwargs:
            v = kwargs[k]
            if _is_jumpy(v):
                kwargs[k] = v.var
        out = f(*args, **kwargs)
        ot = type(out)
        if _is_sdvar(out):
            return SDVariableWrapper(out)
        elif ot in (tuple, list):
            return ot([SDVariableWrapper(o) if _is_sdvar(o) else o for o in out])
        else:
            return out
    return wrapper

class SDVariableWrapper(object):
    def __init__(self, var):
        if _is_jumpy(var):
            self.var = var.var
        elif _is_sdvar(var):
            self.var = var
        else:
            raise Exception("Unsupported type received : " + str(type(var)))

    @op
    def __add__(self, other):
        return self.var.add(other)

    @op
    def __radd__(self, other):
        return self.var.radd(other)

    @op
    def __iadd__(self, other):
        return self.var.addi(other)

    @op
    def __sub__(self, other):
        return self.var.sub(other)

    @op
    def __rsub__(self, other):
        return self.var.rsub(other)

    @op
    def __isub__(self, other):
        return self.var.subi(other)

    @op
    def __mul__(self, other):
        return self.var.mul(other)

    @op
    def __rmul__(self, other):
        return self.var.rmul(other)

    @op
    def __imul__(self, other):
        return self.var.muli(other)

    @op
    def __div__(self, other):
        return self.var.div(other)

    @op
    def __rdiv__(self, other):
        return self.var.rdiv(other)

    def __getattr__(self, attr):
        return getattr(self.var, attr)
