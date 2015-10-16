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

        # Learning rate / step size will change during the iterations, so we'll
        # make it a shared variable.
        if not 'learning_rate' in optimization_options:
            raise ValueError("Learning rate is not given in optimization options.")
        self.param_init_values['optimizer.learning_rate'] = \
            numpy.dtype(theano.config.floatX).type(
                optimization_options['learning_rate'])

        for name, param in network.params.items():
            self.param_init_values[name + '.gradient'] = numpy.zeros_like(param.get_value())

        self._create_params()

        super().__init__(optimization_options, network, *args, **kwargs)

    def _get_gradient_updates(self):
        result = []
        for name, gradient_new in zip(self.network.params, self._gradient_exprs):
            gradient = self.params[name + '.gradient']
            result.append((gradient, gradient_new))
        return result

    def _get_model_updates(self):
        alpha = self.params['optimizer.learning_rate']

        result = []
        for name, param in self.network.params.items():
            gradient = self.params[name + '.gradient']
            result.append((param, param - alpha * gradient))
        return result
