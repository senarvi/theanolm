#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from theano import printing

def print_tensor(message, variable):
    """A small helper function that makes printing Theano variables a little bit
    easier.

    :type message: str
    :param message: message, typically the variable name

    :type variable: TensorVariable
    :param variable: any tensor variable to be printed

    :rtype: TensorVariable
    :returns: a tensor variable to be used further down the graph in place of
              ``variable``
    """

    print_op = printing.Print(message)
    return print_op(variable)
