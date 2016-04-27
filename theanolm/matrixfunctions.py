#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import theano
from theano.compile.debugmode import InvalidValueError

def random_weight(in_size, out_size, scale=None):
    """Generates a weight matrix from “standard normal” distribution.

    :type in_size: int
    :param in_size: size of the input dimension of the weight

    :type out_size: int
    :param out_size: size of the output dimension of the weight

    :type scale: float
    :param scale: if other than None, the matrix will be scaled by this factor

    :rtype: numpy.ndarray
    :returns: the generated weight matrix
    """

    result = numpy.random.randn(in_size, out_size)
    if scale is not None:
        result = scale * result
    return result.astype(theano.config.floatX)

def orthogonal_weight(in_size, out_size, scale=None):
    """Generates a weight matrix from “standard normal” distribution. If in_size
    matches out_size, generates an orthogonal matrix.

    :type in_size: int
    :param in_size: size of the input dimension of the weight

    :type out_size: int
    :param out_size: size of the output dimension of the weight

    :type scale: float
    :param scale: if other than None, the matrix will be scaled by this factor,
                  unless an orthogonal matrix is created
    """

    if in_size != out_size:
        return random_weight(in_size, out_size, scale)

    nonorthogonal_matrix = numpy.random.randn(in_size, out_size)
    result, _, _ = numpy.linalg.svd(nonorthogonal_matrix)
    return result.astype(theano.config.floatX)

def test_value(size, max_value):
    """Creates a matrix of random numbers that can be used as a test value for a
    parameter to enable debugging Theano errors.

    The type of ``max_value`` defines the type of the returned array. For
    integers, the range does not include the maximum value. If ``max_value`` is
    a boolean, returns an int8 array, as Theano uses int8 to represent a
    boolean.

    :type size: int or tuple of ints
    :param size: dimensions of the matrix

    :type max_value: int or float
    :param max_value: maximum value for the generated random numbers

    :rtype: numpy.ndarray
    :returns: a matrix or vector containing the generated values
    """

    if type(max_value) is int:
        return numpy.random.randint(0, max_value, size=size).astype('int64')
    elif type(max_value) is float:
        return max_value * numpy.random.rand(*size).astype(theano.config.floatX)
    elif type(max_value) is bool:
        return numpy.random.randint(0, int(max_value), size=size).astype('int8')
    else:
        raise InvalidValueError("test_value() expects int, float, or bool "
                                "maximum value.")

def get_submatrix(matrices, index, size, end_index=None):
    """Returns a submatrix of a concatenation of 2 or 3 dimensional
    matrices.

    :type matrices: TensorVariable
    :param matrices: symbolic 2 or 3 dimensional matrix formed by
                     concatenating matrices of length size

    :type index: int
    :param index: index of the matrix to be returned

    :type size: TensorVariable
    :param size: size of the last dimension of one submatrix

    :type end_index: int
    :param end_index: if set to other than None, returns a concatenation of all
                      the submatrices from ``index`` to ``end_index``
    """

    if end_index is None:
        end_index = index
    start = index * size
    end = (end_index + 1) * size
    if matrices.ndim == 3:
        return matrices[:, :, start:end]
    elif matrices.ndim == 2:
        return matrices[:, start:end]
    else:
        raise ValueError("get_submatrix() requires a 2 or 3 dimensional matrix.")
