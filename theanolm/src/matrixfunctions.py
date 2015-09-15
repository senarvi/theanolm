#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy

def random_weight(in_size, out_size, scale=None):
	""" Generates a weight matrix from “standard normal” distribution.

	:type scale: float
	:param scale: if other than None, the matrix will be scaled by this factor
	"""

	result = numpy.random.randn(in_size, out_size)
	if scale is not None:
		result = scale * result
	return result.astype('float32')

def orthogonal_weight(in_size, out_size, scale=None):
	""" Generates a weight matrix from “standard normal” distribution. If
	in_size matches out_size, generates an orthogonal matrix.

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
	return result.astype('float32')

def get_submatrix(matrices, index, size):
	"""Returns a submatrix of a concatenation of 2 or 3 dimensional
	matrices.

	:type matrices: theano.tensor.var.TensorVariable
	:param matrices: symbolic 2 or 3 dimensional matrix formed by
	                 concatenating matrices of length size

	:type index: int
	:param index: index of the matrix to be returned

	:type size: theano.tensor.var.TensorVariable
	:param size: size of the last dimension of one submatrix
	"""

	start = index * size
	end = (index + 1) * size
	if matrices.ndim == 3:
		return matrices[:,:,start:end]
	elif matrices.ndim == 2:
		return matrices[:,start:end]
	else:
		raise ValueError("get_submatrix() requires a 2 or 3 dimensional matrix.")
