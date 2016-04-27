#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import theano
from theanolm.optimizers.basicoptimizer import BasicOptimizer

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

        self.param_init_values = dict()
        for name, param in network.params.items():
            self.param_init_values[name + '_gradient'] = \
                numpy.zeros_like(param.get_value())

        super().__init__(optimization_options, network, *args, **kwargs)

    def _gradient_update_exprs(self):
        result = []
        for name, gradient_new in zip(self.network.params,
                                      self._gradient_exprs):
            gradient = self.params[name + '_gradient']
            result.append((gradient, gradient_new))
        return result

    def _model_update_exprs(self, alpha):
        updates = dict()
        for name, param in self.network.params.items():
            gradient = self.params[name + '_gradient']
            updates[name] = -gradient
        self._normalize(updates)

        result = []
        for name, param in self.network.params.items():
            update = updates[name]
            result.append((param, param + alpha * update))
        return result
