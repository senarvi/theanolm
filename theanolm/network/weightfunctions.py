#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import theano

def random_weight_matrix(shape, scale=None):
    """Generates a weight matrix from “standard normal” distribution.

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

def orthogonal_weight_matrix(size):
    """Generates an orthogonal weight matrix from “standard normal”
    distribution.

    :type size: int
    :param size: size of both dimensions of the weight

    :rtype: numpy.ndarray
    :returns: the generated weight matrix
    """

    nonorthogonal_matrix = numpy.random.randn(size, size)
    result, _, _ = numpy.linalg.svd(nonorthogonal_matrix)
    return result.astype(theano.config.floatX)

def weight_matrix(shape, scale=None, count=1):
    """Generates a weight matrix from “standard normal” distribution.

    If ``shape`` contains two dimensions that match, generates an orthogonal
    matrix. In that case scale is ignored. Orthogonal weights are useful for two
    reasons:

    1. Multiplying by an orthogonal weight preserves the norm of the input
       vector, which should help avoid exploding and vanishing gradients.
    2. The row and column vectors are orthonormal to one another, which should
       help avoid two vectors learning to produce the same features.

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
            [orthogonal_weight_matrix(shape[0]) for _ in range(count)],
            axis=1)
    else:
        return numpy.concatenate(
            [random_weight_matrix(shape, scale) for _ in range(count)],
            axis=1)


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
