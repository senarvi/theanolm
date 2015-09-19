#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class InputError(Exception):
	"""Exception raised for errors in the input.
	"""
	pass

class NumberError(Exception):
	"""Exception raised when one of the parameter gets NaN value.
	"""
	pass
