#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functions for debugging.
"""

from theano import tensor, printing

def print_tensor(message, variable):
    """A small helper function that makes printing Theano variables a little bit
    easier.

    :type message: str
    :param message: message, typically the variable name

    :type variable: Variable
    :param variable: any tensor variable to be printed

    :rtype: Variable
    :returns: a tensor variable to be used further down the graph in place of
              ``variable``
    """

    print_op = printing.Print(message)
    return print_op(variable)

def assert_tensor_eq(result, name1, name2, variable1, variable2):
    """A small helper function that makes it a little bit easier to assert that
    two Theano variables are equal.

    :type result: Variable
    :param result: what the result of the operation should be

    :type name1: str
    :param name1: name of the first variable

    :type name2: str
    :param name2: name of the second variable

    :type variable1: Variable
    :param variable1: the first variable

    :type variable2: Variable
    :param variable2: the second variable

    :rtype: Variable
    :returns: a tensor variable that returns the same value as ``result``, and
              asserts that ``variable1`` equals to ``variable2``
    """

#    print_op = printing.Print(name1 + ":")
#    variable1 = tensor.switch(tensor.neq(variable1, variable2),
#                              print_op(variable1),
#                              variable1)
#    print_op = printing.Print(name2 + ":")
#    variable2 = tensor.switch(tensor.neq(variable1, variable2),
#                              print_op(variable2),
#                              variable2)
    assert_op = tensor.opt.Assert(name1 + " != " + name2)
    return assert_op(result, tensor.eq(variable1, variable2))
