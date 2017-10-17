#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the Adam optimizer.
"""

import numpy
import theano
import theano.tensor as tensor

from theanolm.backend import Parameters
from theanolm.training.basicoptimizer import BasicOptimizer

class AdamOptimizer(BasicOptimizer):
    """Adam Optimization Method

    D. P. Kingma, J. Ba (2015)
    Adam: A Method for Stochastic Optimization
    The International Conference on Learning Representations (ICLR), San Diego
    """

    def __init__(self, optimization_options, network, *args, **kwargs):
        """Creates an Adam optimizer.

        :type optimization_options: dict
        :param optimization_options: a dictionary of optimization options

        :type network: Network
        :param network: the neural network object
        """

        self._params = Parameters()

        float_type = numpy.dtype(theano.config.floatX).type
        self._params.add('optimizer/timestep', float_type(0.0))

        for path, param in network.get_variables().items():
            self._params.add(path + '_mean_gradient',
                             numpy.zeros_like(param.get_value()))
            self._params.add(path + '_mean_sqr_gradient',
                             numpy.zeros_like(param.get_value()))

        # geometric rate for averaging gradients
        if 'gradient_decay_rate' not in optimization_options:
            raise ValueError("Gradient decay rate is not given in training "
                             "options.")
        self._gamma_m = optimization_options['gradient_decay_rate']

        # geometric rate for averaging squared gradients
        if 'sqr_gradient_decay_rate' not in optimization_options:
            raise ValueError("Squared gradient decay rate is not given in "
                             "optimization options.")
        self._gamma_ms = optimization_options['sqr_gradient_decay_rate']

        # momentum
        if 'momentum' not in optimization_options:
            raise ValueError("Momentum is not given in optimization options.")
        self._momentum = optimization_options['momentum']

        super().__init__(optimization_options, network, *args, **kwargs)

    def _get_param_updates(self, alpha):
        """Returns Theano expressions for updating the model parameters and any
        additional parameters required by the optimizer.

        :type alpha: Variable
        :param alpha: a scale to be applied to the model parameter updates

        :rtype: iterable over pairs (shared variable, new expression)
        :returns: expressions how to update the optimizer parameters
        """

        timestep_old = self._params['optimizer/timestep']
        timestep = timestep_old + 1.0
        alpha *= tensor.sqrt(1.0 - (self._gamma_ms ** timestep))
        alpha /= 1.0 - (self._gamma_m ** timestep)

        result = []
        deltas = dict()
        for path, gradient in zip(self.network.get_variables(),
                                  self._gradients):
            m_gradient_old = self._params[path + '_mean_gradient']
            ms_gradient_old = self._params[path + '_mean_sqr_gradient']
            m_gradient = \
                self._gamma_m * m_gradient_old + \
                (1.0 - self._gamma_m) * gradient
            ms_gradient = \
                self._gamma_ms * ms_gradient_old + \
                (1.0 - self._gamma_ms) * tensor.sqr(gradient)
            result.append((m_gradient_old, m_gradient))
            result.append((ms_gradient_old, ms_gradient))

            rms_gradient = tensor.sqrt(ms_gradient) + self._epsilon
            deltas[path] = -m_gradient / rms_gradient
        self._normalize(deltas)

        for path, param_old in self.network.get_variables().items():
            delta = deltas[path]
            result.append((param_old, param_old + alpha * delta))
        result.append((timestep_old, timestep))
        return result
