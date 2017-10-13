#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that extends the operations provided by Theano.

The functions are designed to be compatible with the Keras interface.
"""

import theano.tensor as tensor
from theano.gof import MissingInputError

def conv2d(input_matrix, filters, strides=(1, 1), padding='valid'):
    """Performs 2-dimensional convolution given 4-dimensional input and filter
    tensors.

    The input is a mini-batch of 2-dimensional n-channel samples. Convolution is
    applied independently to each sample.

    The operation can add padding to the input tensor in order apply the filter
    also in places where it does not completely overlap with the input. At least
    the following modes are supported:

      - ``same``: Apply so much that the output dimensions will be the same as
                  the input dimensions.
      - ``valid``: Only apply the filter where all elements overlap with the
                   input. The output will be smaller than the input.
      - ``full``: Apply the filter where any part of it overlaps with the input.
                  The output will be larger than the input.

    :type input_matrix: symbolic 4D tensor
    :param input_matrix: a mini-batch of shape (samples, rows, columns, features)

    :type filters: symbolic 3D tensor
    :param filters: filters of shape (rows, columns, input features, output
                    features)

    :type strides: int
    :param strides: the steps in which the sliding window is shifted in each
                    dimension

    :type padding: str
    :param padding: padding to be applied to the input
    """

    # Permute input dimensions from (samples, rows, columns, features) to
    # (samples, features, rows, columns).
    input_matrix = input_matrix.dimshuffle(0, 3, 1, 2)
    # Permute filter dimensions from (rows, columns, input features, output
    # features) to (output features, input features, rows, columns)`.
    filters = filters.dimshuffle(3, 2, 0, 1)
    # Convert padding to border_mode values understood by Theano.
    if padding == 'same':
        border_mode = 'half'
    elif padding == 'valid':
        border_mode = 'valid'
    elif padding == 'full':
        border_mode = 'full'
    else:
        raise ValueError('Unknown input padding requested: {}'.format(padding))
    # If filters is a shared variable, we can get its shape and use it to select
    # an optimal implementation.
    try:
        filter_shape = filters.eval().shape
    except MissingInputError:
        filter_shape = None

    result = tensor.nnet.conv2d(input_matrix,
                                filters,
                                border_mode=border_mode,
                                subsample=strides,
                                input_shape=None,
                                filter_shape=filter_shape,
                                filter_flip=False)

    if padding == 'same':
        # If the filter width or height is not an odd number, there will be an
        # extra element in the output.
        rows_end = (input_matrix.shape[2] + strides[0] - 1) // strides[0]
        result = result[:, :, :rows_end, :]
        columns_end = (input_matrix.shape[3] + strides[1] - 1) // strides[1]
        result = result[:, :, :, :columns_end]

    # Permute output dimensions from (samples, features, rows, columns) to
    # (samples, rows, columns, features).
    return result.dimshuffle(0, 2, 3, 1)

def conv1d(input_matrix, filters, strides=1, padding='valid'):
    """Performs 1-dimensional convolution given 3-dimensional input and filter
    tensors.

    The input is a mini-batch of 1-dimensional n-channel samples. Convolution is
    applied independently to each sample.

    Adds a dimension of size 1 to input and filter tensors and calls `conv2d`.
    Compatible with the Keras interface.

    The operation can add padding to the input tensor in order apply the filters
    also in places where it does not completely overlap with the input. At least
    the following modes are supported:

      - ``same``: Apply so much that the output dimensions will be the same as
                  the input dimensions.
      - ``valid``: Only apply the filters where all elements overlap with the
                   input. The output will be smaller than the input.
      - ``full``: Apply the filters where any part of it overlaps with the input.
                  The output will be larger than the input.

    :type input_matrix: symbolic 3D tensor
    :param input_matrix: a mini-batch of shape (samples, elements, features)

    :type filters: symbolic 3D tensor
    :param filters: filters of shape (elements, input features, output features)

    :type strides: int
    :param strides: the steps in which the sliding window is shifted

    :type padding: str
    :param padding: padding to be applied to the input
    """

    # Permute input dimensions from (samples, elements, features) to
    # (samples, rows, 1, features).
    input_matrix = input_matrix.dimshuffle(0, 1, 'x', 2)
    # Permute filter dimensions from (elements, input features, output features)
    # to (rows, 1, input features, output features).
    filters = filters.dimshuffle(0, 'x', 1, 2)
    # Stride dimensions from (strides) to (strides, 1).
    strides = (strides, 1)

    output = conv2d(input_matrix, filters, strides=strides, padding=padding)

    # Output dimensions from (samples, rows, 1, features) to
    # (samples, elements, features).
    return output.reshape([output.shape[0], output.shape[1], output.shape[3]])
