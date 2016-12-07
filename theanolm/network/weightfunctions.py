#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import theano

def random_normal_matrix(shape, scale=None):
    """Generates a random matrix from “standard normal” distribution.

    :type shape: tuple of ints
    :param shape: size of each dimension (typically there are two dimensions,
                  input and output)

    :type scale: float
    :param scale: if other than None, the random numbers will be scaled by this
                  factor

    :rtype: numpy.ndarray
    :returns: the generated weight matrix
    """

    result = numpy.random.randn(*shape)
    if scale is not None:
        result = scale * result
    return result.astype(theano.config.floatX)

def random_orthogonal_matrix(size):
    """Generates a random orthogonal matrix from “standard normal” distribution.

    :type size: int
    :param size: size of both dimensions of the weight

    :rtype: numpy.ndarray
    :returns: the generated weight matrix
    """

    nonorthogonal_matrix = numpy.random.randn(size, size)
    result, _, _ = numpy.linalg.svd(nonorthogonal_matrix)
    return result.astype(theano.config.floatX)

def random_matrix(shape, scale=None, count=1):
    """Generates a random matrix from “standard normal” distribution.

    If ``shape`` contains two dimensions that match, generates an orthogonal
    matrix. In that case scale is ignored.

    If ``count`` is specified, creates a concatenation of several similar
    matrices (same shape but different content).

    :type shape: tuple of ints
    :param shape: size of each dimension (typically there are two dimensions,
                  input and output)

    :type scale: float
    :param scale: if other than None, the random numbers will be scaled by this
                  factor

    :type count: int
    :param count: concatenate this many weight matrices with the same shape

    :rtype: numpy.ndarray
    :returns: the generated weight matrix
    """

    if (len(shape) == 2) and (shape[0] == shape[1]):
        return numpy.concatenate(
            [random_orthogonal_matrix(shape[0]) for _ in range(count)],
            axis=-1)
    else:
        return numpy.concatenate(
            [random_normal_matrix(shape, scale) for _ in range(count)],
            axis=-1)

def matrix_from_value(shape, value=None):
    """Creates a matrix with given value.

    If ``value`` is not given, initializes the vector with zero value. If
    ``value`` is a list, creates a concatenation of as many vectors as there are
    elements in the list.

    :type shape: int or tuple of ints
    :param shape: size of the vector, or a tuple of the sizes of each dimension
                  (in case ``value`` is a list, each part will have this size)

    :type value: float, numpy.ndarray or list
    :param value: the value or array to initialize the elements to, or a list of
                  values or arrays to create a concatenation of vectors
    """

    values = value if isinstance(value, (list, tuple)) else [value]
    parts = []
    for part_value in values:
        if part_value is None:
            part = numpy.zeros(shape).astype(theano.config.floatX)
        elif isinstance(value, numpy.ndarray):
            part = value.astype(theano.config.floatX)
        else:
            part = numpy.empty(shape).astype(theano.config.floatX)
            part.fill(part_value)
        parts.append(part)
    return numpy.concatenate(parts, axis=-1)

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
