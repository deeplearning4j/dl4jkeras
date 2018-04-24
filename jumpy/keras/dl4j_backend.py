from keras.backend.common import floatx, epsilon
from keras.backend.common import image_data_format

from ..java_classes import SameDiff, Nd4j
from ..ndarray import array, get_context_dtype
from ..matlib import zeros as nd4j_zeros
from ..matlib import ones as nd4j_ones

from .samediff import SDVariableWrapper


sd = SameDiff.create()

def _to_int_shape(x):
    if x is None:
        return -1
    else:
        return [-1 if i is None else i for i in x]

opnames = set()
def get_opname(op):
    i = 0
    while(True):
        opname = op + '_' + str(i)
        if opname not in opnames:
            opnames.add(opname)
            return opname
        i += 1


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    if shape is None:
        if ndim is None:
            shape = None
        else:
            shape = [None for _ in range(ndim)]
    ph = SDVariableWrapper(sd.var(get_opname("placeholder"), _to_int_shape(shape)))
    ph.placeholder = True
    ph._keras_shape = shape
    ph._uses_learning_phase = False
    return ph


def is_placeholder(x):
    return getattr(x, 'placeholder', False)


def variable(value, dtype=None, name=None, constraint=None):
    value = array(value).array  # INDArray
    var = SDVariableWrapper(sd.var(get_opname("variable"), value))
    var._keras_shape = var.shape
    var._uses_learning_phase = False
    return var


def constant(value, dtype=None, shape=None, name=None):
    return variable(value)


def is_keras_tensor(x):
    return hasattr(x, '_keras_history')


def shape(x):
    s = x.shape
    if x is None:
        s = x.add(0).eval().shape()
    fs = [float(x) for x in s]
    return variable(fs)


def int_shape(x):
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    try:
        s = x.shape
        if s is None:
            s = x.add(0).eval().shape()
        return [None if i < 0 else i for i in s]
    except:
        return None


def ndim(x):
    s = int_shape(x)
    if s is None:
        return None
    else:
        return len(s)


def dtype(x):
    return get_context_dtype()


def eval(x):
    return array(x.add(0).eval())


def zeros(shape, dtype=None, name=None):
    return variable(nd4j_zeros(shape))


def ones(shape, dtype=None, name=None):
    return variable(nd4j_ones(shape))


def eye(size, dtype=None, name=None):
    return variable(Nd4j.eye(size))

@op
def zeros_like(x, dtype=None, name=None):
    return sd.zerosLike(x)

@op
def ones_like(x, dtype=None, name=None):
    return sd.onesLike(x)

@op
def identity(x,name=None):
    return x.add(0)


def random_uniform_variable(shape, low, high, dtype=None,
                            name=None, seed=None):
    raise NotImplemented


def random_normal_variable(shape, mean, scale, dtype=None,
                           name=None, seed=None):
    raise NotImplemented


def cast(x, dtype):
    raise NotImplemented


def count_params(x):
    p = 1
    s = int_shape(x)
    for i in s:
        p *= i
    return p

@op
def update(x, new_x):
    """Update the value of `x` to `new_x`.

    # Arguments
        x: A `Variable`.
        new_x: A tensor of same shape as `x`.

    # Returns
        The variable `x` updated.
    """
    return sd.assign(x, new_x)

@op
def update_add(x, increment):
    """Update the value of `x` by adding `increment`.

    # Arguments
        x: A `Variable`.
        increment: A tensor of same shape as `x`.

    # Returns
        The variable `x` updated.
    """
    return x.addi(increment)

@op
def update_sub(x, decrement):
    """Update the value of `x` by subtracting `decrement`.

    # Arguments
        x: A `Variable`.
        decrement: A tensor of same shape as `x`.

    # Returns
        The variable `x` updated.
    """
    return x.subi(decrement)


def moving_average_update(x, value, momentum):
    raise NotImplemented

@op
def dot(x, y):
    return sd.mmul(x, y)

@op
def batch_dot(x, y, axes=None):
    if isinstance(axes, int):
        axes = [[axis], [axis]]
    if axes is None:
        # behaves like tf.batch_matmul as default
        axes = [[x_ndim - 1], [y_ndim - 2]]
    axes = [[a] if type(a) is int else a for a in axes]
    return sd.tensorMmul(x, y, axes)

@op
def transpose(x):
    return sd.transpose(x)

@op
def gather(reference, indices):
    """Retrieves the elements of indices `indices` in the tensor `reference`.

    # Arguments
        reference: A tensor.
        indices: An integer tensor of indices.

    # Returns
        A tensor of same type as `reference`.
    """
    return sd.gatherNd(reference, indices)


# ELEMENT-WISE OPERATIONS

@op
def max(x, axis=None, keepdims=False):
    """Maximum value in a tensor.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to find maximum values.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with maximum values of `x`.
    """
    if axis is None:
        if keepdims:
            return sd.reshape(sd.max(x), * [1] * ndim(x))
        else:
            return sd.max(x)
    mx = sd.max(x, axis)
    if keepdims:
        mx = sd.expandDims(mx, axis)
    return mx

@op
def min(x, axis=None, keepdims=False):
    """Minimum value in a tensor.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to find minimum values.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with miminum values of `x`.
    """
    if axis is None:
        if keepdims:
            return sd.reshape(sd.min(x), * [1] * ndim(x))
        else:
            return sd.min(x)
    mn = sd.min(x, axis)
    if keepdims:
        mn = sd.expandDims(mn, axis)
    return mn

@op
def sum(x, axis=None, keepdims=False):
    """Sum of the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to sum over.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with sum of `x`.
    """
    if axis is None:
        if keepdims:
            return sd.reshape(sd.sum(x), * [1] * ndim(x))
        else:
            return sd.sum(x)
    s = sd.sum(x, axis)
    if keepdims:
        s = sd.expandDims(s, axis)
    return s

@op
def prod(x, axis=None, keepdims=False):
    """Multiplies the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the product.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the product of elements of `x`.
    """
    if axis is None:
        if keepdims:
            return sd.reshape(sd.prod(x), * [1] * ndim(x))
        else:
            return sd.prod(x)
    p = sd.prod(x, axis)
    if keepdims:
        p = sd.expandDims(p, axis)
    return p


@op
def cumsum(x, axis=0):
    """Cumulative sum of the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the sum.

    # Returns
        A tensor of the cumulative sum of values of `x` along `axis`.
    """
    return sd.cumsum(x, False, False, axis)

@op
def cumprod(x, axis=0):
    """Cumulative product of the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the product.

    # Returns
        A tensor of the cumulative product of values of `x` along `axis`.
    """
    return sd.cumprod(x, False, False, axis)

@op
def var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """

    if axis is None:
        axis = [i for i in range(ndim(x))]
    elif type(axis) is int:
        axis = [axis]
    v = sd.variance(x, False, *axis)
    if keepdims:
        v = sd.expandDims(v, axis)
    return v

@op
def std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    if axis is None:
        axis = [i for i in range(ndim(x))]
    elif type(axis) is int:
        axis = [axis]
    v = sd.standardDeviation(x, False, *axis)
    if keepdims:
        v = sd.expandDims(v, axis)
    return v

@op
def mean(x, axis=None, keepdims=False):
    """Mean of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: A list of integer. Axes to compute the mean.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1 for each entry in `axis`. If `keepdims` is `True`,
            the reduced dimensions are retained with length 1.

    # Returns
        A tensor with the mean of elements of `x`.
    """
    if axis is None:
        m = sd.mean(x)
        if keepdims:
            return sd.reshape(m, * [1] * ndim(x))
        return m
    m = sd.mean(x, axis)
    if keepdims:
        m = sd.expandDims(m, axis)
    return m

@op
def any(x, axis=None, keepdims=False):
    """Bitwise reduction (logical OR).

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform the reduction.
        keepdims: whether the drop or broadcast the reduction axes.

    # Returns
        A uint8 tensor (0s and 1s).
    """
    raise NotImplemented

@op
def all(x, axis=None, keepdims=False):
    """Bitwise reduction (logical AND).

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform the reduction.
        keepdims: whether the drop or broadcast the reduction axes.

    # Returns
        A uint8 tensor (0s and 1s).
    """
    raise NotImplemented

@op
def argmax(x, axis=-1):
    """Returns the index of the maximum value along an axis.

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform the reduction.

    # Returns
        A tensor.
    """
    if axis < 0:
        axis += ndim(x)
    return sd.argmax(x, axis)

@op
def argmin(x, axis=-1):
    """Returns the index of the minimum value along an axis.

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform the reduction.

    # Returns
        A tensor.
    """
    if axis < 0:
        axis += ndim(x)
    return sd.argmax(x, axis)

@op
def square(x):
    """Element-wise square.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return sd.square(x)

@op
def abs(x):
    """Element-wise absolute value.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return sd.abs(x)

@op
def sqrt(x):
    """Element-wise square root.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return sd.sqrt(x)


@op
def exp(x):
    """Element-wise exponential.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return sd.exp(x)

@op
def log(x):
    """Element-wise log.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return sd.log(x)

@op
def logsumexp(x, axis=None, keepdims=False):
    """Computes log(sum(exp(elements across dimensions of a tensor))).

    This function is more numerically stable than log(sum(exp(x))).
    It avoids overflows caused by taking the exp of large inputs and
    underflows caused by taking the log of small inputs.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to reduce over.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`, the reduced dimension is
            retained with length 1.

    # Returns
        The reduced tensor.
    """
    raise NotImplemented

@op
def round(x):
    """Element-wise rounding to the closest integer.

    In case of tie, the rounding mode used is "half to even".

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return sd.round(x)

@op
def sign(x):
    """Element-wise sign.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return sd.sign(x)


@op
def pow(x, a):
    """Element-wise exponentiation.

    # Arguments
        x: Tensor or variable.
        a: Python integer.

    # Returns
        A tensor.
    """
    return sd.pow(x, a)

@op
def clip(x, min_value, max_value):
    """Element-wise value clipping.

    # Arguments
        x: Tensor or variable.
        min_value: Python float or integer.
        max_value: Python float or integer.

    # Returns
        A tensor.
    """
    if max_value is not None and max_value < min_value:
        max_value = min_value
    if max_value is None:
        max_value = np.inf
    return sd.clipByValue(x, min_value, max_value)

@op
def equal(x, y):
    """Element-wise equality between two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    return sd.eq(x, y)

@op
def not_equal(x, y):
    """Element-wise inequality between two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    return sd.neq(x, y)

@op
def greater(x, y):
    """Element-wise truth value of (x > y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    return sd.gt(x, y)

@op
def greater_equal(x, y):
    """Element-wise truth value of (x >= y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    return sd.gte(x, y)

@op
def less(x, y):
    """Element-wise truth value of (x < y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    return sd.lt(x, y)

@op
def less_equal(x, y):
    """Element-wise truth value of (x <= y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    return sd.lte(x, y)

@op
def maximum(x, y):
    """Element-wise maximum of two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor.
    """
    return sd.max(x, y)

@op
def minimum(x, y):
    """Element-wise minimum of two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor.
    """
    return sd.min(x, y)

@op
def sin(x):
    """Computes sin of x element-wise.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return sd.sin(x)

@op
def cos(x):
    """Computes cos of x element-wise.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return sd.cos(x)

@op
def normalize_batch_in_training(x, gamma, beta,
                                reduction_axes, epsilon=1e-3):
    raise NotImplemented


@op
def batch_normalization(x, mean, var, beta, gamma, epsilon=1e-3):
    """Applies batch normalization on x given mean, var, beta and gamma.

    I.e. returns:
    `output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta`

    # Arguments
        x: Input tensor or variable.
        mean: Mean of batch.
        var: Variance of batch.
        beta: Tensor with which to center the input.
        gamma: Tensor by which to scale the input.
        epsilon: Fuzz factor.

    # Returns
        A tensor.
    """
    return sd.batcNorm(x, mean, var, gamma, beta)

@op
def concatenate(tensors, axis=-1):
    if axis < 0:
        axis += ndim(tensors[0])
    return sd.concat(axis, *tensors)

@op
def reshape(x, shape):
    """Reshapes a tensor to the specified shape.

    # Arguments
        x: Tensor or variable.
        shape: Target shape tuple.

    # Returns
        A tensor.
    """
    return sd.reshape(x, *shape)

@op
def permute_dimensions(x, pattern):
    """Permutes axes in a tensor.

    # Arguments
        x: Tensor or variable.
        pattern: A tuple of
            dimension indices, e.g. `(0, 2, 1)`.

    # Returns
        A tensor.
    """
    return sd.permute(x, *pattern)

@op
def resize_images(x, height_factor, width_factor, data_format):
    raise NotImplemented

@op
def resize_volumes(x, depth_factor, height_factor, width_factor, data_format):
    raise NotImplemented

@op
def repeat_elements(x, rep, axis):
    """Repeats the elements of a tensor along an axis, like `np.repeat`.

    If `x` has shape `(s1, s2, s3)` and `axis` is `1`, the output
    will have shape `(s1, s2 * rep, s3)`.

    # Arguments
        x: Tensor or variable.
        rep: Python integer, number of times to repeat.
        axis: Axis along which to repeat.

    # Returns
        A tensor.
    """
    for _ in range(rep):
        x = sd.repeat(x, axis)
    return x

@op
def repeat(x, n):
    """Repeats a 2D tensor.

    if `x` has shape (samples, dim) and `n` is `2`,
    the output will have shape `(samples, 2, dim)`.

    # Arguments
        x: Tensor or variable.
        n: Python integer, number of times to repeat.

    # Returns
        A tensor.
    """
    assert ndim(x) == 2
    x = repeat_elements(x, n, 1)
    new_shape = list(int_shape(x))
    new_shape.insert(1, n)
    x = sd.reshape(x, *new_shape)

@op
def arange(start, stop=None, step=1, dtype='int32'):
    """Creates a 1D tensor containing a sequence of integers.

    The function arguments use the same convention as
    Theano's arange: if only one argument is provided,
    it is in fact the "stop" argument and "start" is 0.

    The default type of the returned tensor is `'int32'` to
    match TensorFlow's default.

    # Arguments
        start: Start value.
        stop: Stop value.
        step: Difference between two successive values.
        dtype: Integer dtype to use.

    # Returns
        An integer tensor.

    """
    raise NotImplemented


@op
def tile(x, n):
    """Creates a tensor by tiling `x` by `n`.

    # Arguments
        x: A tensor or variable
        n: A list of integer. The length must be the same as the number of
            dimensions in `x`.

    # Returns
        A tiled tensor.
    """
    if isinstance(n, int):
        n = [n]
    return sd.tile(x, n)

@op
def flatten(x):
    """Flatten a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor, reshaped into 1-D
    """
    s = int_shape(x)
    vec_size = 1
    for i in s:
        vec_size *= i
    return sd.reshape(x, vec_size)

@op
def batch_flatten(x):
    s = int_shape(x)
    vec_size = 1
    for i in s[1:]:
        vec_size *= i
    return sd.reshape(x, -1, vec_size)

@op
def expand_dims(x, axis=-1):
    """Adds a 1-sized dimension at index "axis".

    # Arguments
        x: A tensor or variable.
        axis: Position where to add a new axis.

    # Returns
        A tensor with expanded dimensions.
    """
    return sd.expandDims(x, axis)

@op
def squeeze(x, axis):
    """Removes a 1-dimension from the tensor at index "axis".

    # Arguments
        x: A tensor or variable.
        axis: Axis to drop.

    # Returns
        A tensor with the same data as `x` but reduced dimensions.
    """
    return sd.squeeze(x, axis)

@op
def temporal_padding(x, padding=(1, 1)):
    raise NotImplemented


@op
def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    raise NotImplemented

@op
def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
    raise NotImplemented

@op
def stack(x, axis=0):
    """Stacks a list of rank `R` tensors into a rank `R+1` tensor.

    # Arguments
        x: List of tensors.
        axis: Axis along which to perform stacking.

    # Returns
        A tensor.
    """
    return tf.stack(x, axis)


@op
def one_hot(indices, num_classes):
    """Computes the one-hot representation of an integer tensor.

    # Arguments
        indices: nD integer tensor of shape
            `(batch_size, dim1, dim2, ... dim(n-1))`
        num_classes: Integer, number of classes to consider.

    # Returns
        (n + 1)D one hot representation of the input
        with shape `(batch_size, dim1, dim2, ... dim(n-1), num_classes)`
    """
    return sd.oneHot(indices, depth=num_classes)


@op
def reverse(x, axes):
    """Reverse a tensor along the specified axes.

    # Arguments
        x: Tensor to reverse.
        axes: Integer or iterable of integers.
            Axes to reverse.

    # Returns
        A tensor.
    """
    if isinstance(axes, int):
        axes = [axes]
    return tf.reverse(x, *axes)


###-----------NO OPS BEYOND THIS LINE---------------###


def function(inputs, outputs, updates=None, **kwargs):

    class Function(object):
        def __init__(self, inputs, outputs):
            self.inputs = inputs
            self.outputs = outputs

        def __call__(self, X):
            for i, x in zip(self.inputs, X):
                i.associateArrayWithVariable(array(x).array)
            return [array(o.eval()) for o in self.outputs]

    return Function(inputs, outputs)

