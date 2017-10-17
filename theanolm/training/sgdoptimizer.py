#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the Stochastic Gradient Descent optimizer.
"""

import numpy

from theanolm.backend import Parameters
from theanolm.training.basicoptimizer import BasicOptimizer

class SGDOptimizer(BasicOptimizer):
    """Stochastic Gradient Descent Optimization Method
    """

    def __init__(self, optimization_options, network, *args, **kwargs):
        """Creates a Stochastic Gradient Descent optimizer. SGD optimizer does
        not use additional parameters.

        :type optimization_options: dict
        :param optimization_options: a dictionary of optimization options

        :type network: Network
        :param network: the neural network object
        """

        self._params = Parameters()

        super().__init__(optimization_options, network, *args, **kwargs)

    def _get_param_updates(self, alpha):
        """Returns Theano expressions for updating the model parameters and any
        additional parameters required by the optimizer.

        :type alpha: Variable
        :param alpha: a scale to be applied to the model parameter updates

        :rtype: iterable over pairs (shared variable, new expression)
        :returns: expressions how to update the optimizer parameters
        """

        deltas = dict()
        result = []
        for path, gradient in zip(self.network.get_variables(),
                                  self._gradients):
            deltas[path] = -gradient
        self._normalize(deltas)

        for path, param_old in self.network.get_variables().items():
            delta = deltas[path]
            result.append((param_old, param_old + alpha * delta))
        return result
