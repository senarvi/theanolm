#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import theano

def test_value(size, high):
    """Creates a matrix of random numbers that can be used as a test value for a
    parameter to enable debugging Theano errors.

    The type of ``high`` defines the type of the returned array. For integers,
    the range does not include the maximum value. If ``high`` is a boolean,
    returns an int8 array, as Theano uses int8 to represent a boolean.

    :type size: int or tuple of ints
    :param size: dimensions of the matrix

    :type high: int, float, or bool
    :param high: maximum value for the generated random numbers

    :rtype: numpy.ndarray
    :returns: a matrix or vector containing the generated values
    """

    if isinstance(high, bool):
        return numpy.random.randint(0, int(high), size=size).astype('int8')
    elif isinstance(high, (int, numpy.int32, numpy.int64)):
        return numpy.random.randint(0, high, size=size).astype('int64')
    elif isinstance(high, (float, numpy.float32, numpy.float64)):
        return high * numpy.random.rand(*size).astype(theano.config.floatX)
    else:
        raise TypeError("High value should be int, float, or bool.")
