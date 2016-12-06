#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
from theanolm import Parameters
from theanolm.training.basicoptimizer import BasicOptimizer

class SGDOptimizer(BasicOptimizer):
    """Stochastic Gradient Descent Optimization Method
    """

    def __init__(self, optimization_options, network, *args, **kwargs):
        """Creates a Stochastic Gradient Descent optimizer.

        :type optimization_options: dict
        :param optimization_options: a dictionary of optimization options

        :type network: Network
        :param network: the neural network object
        """

        self._params = Parameters()
        for path, param in network.get_variables().items():
            self._params.add(path + '_gradient',
                             numpy.zeros_like(param.get_value()))

        super().__init__(optimization_options, network, *args, **kwargs)

    def _gradient_update_exprs(self):
        result = []
        for path, gradient_new in zip(self.network.get_variables(),
                                      self._gradient_exprs):
            gradient = self._params[path + '_gradient']
            result.append((gradient, gradient_new))
        return result

    def _model_update_exprs(self, alpha):
        updates = dict()
        for path, param in self.network.get_variables().items():
            gradient = self._params[path + '_gradient']
            updates[path] = -gradient
        self._normalize(updates)

        result = []
        for path, param in self.network.get_variables().items():
            update = updates[path]
            result.append((param, param + alpha * update))
        return result
