#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from theano import tensor, printing

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

def assert_tensor_eq(result, name1, name2, variable1, variable2):
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
