#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the ADADELTA optimizer.
"""

import numpy
import theano.tensor as tensor

from theanolm.backend import Parameters
from theanolm.training.basicoptimizer import BasicOptimizer

class AdadeltaOptimizer(BasicOptimizer):
    """ADADELTA Optimization Method

    ADADELTA optimization method has been derived from AdaGrad. AdaGrad
    accumulates the sum of squared gradients over all time, which is used to
    scale the learning rate smaller and smaller. ADADELTA uses an exponentially
    decaying average of the squared gradients.

    This implementation scales the parameter updates by the learning rate
    hyperparameter. The original paper does not include such scaling,
    corresponding to learning rate 1.

    M. D. Zeiler (2012)
    ADADELTA: An adaptive learning rate method
    http://arxiv.org/abs/1212.5701
    """

    def __init__(self, optimization_options, network, *args, **kwargs):
        """Creates an Adadelta optimizer.

        :type optimization_options: dict
        :param optimization_options: a dictionary of optimization options

        :type network: Network
        :param network: the neural network object
        """

        self._params = Parameters()
        for path, param in network.get_variables().items():
            self._params.add(path + '_mean_sqr_gradient',
                             numpy.zeros_like(param.get_value()))
            self._params.add(path + '_mean_sqr_velocity',
                             numpy.zeros_like(param.get_value()))

        # geometric rate for averaging gradients
        if 'gradient_decay_rate' not in optimization_options:
            raise ValueError("Gradient decay rate is not given in optimization "
                             "options.")
        self._gamma = optimization_options['gradient_decay_rate']

        super().__init__(optimization_options, network, *args, **kwargs)

    def _get_param_updates(self, alpha):
        """Returns Theano expressions for updating the model parameters and any
        additional parameters required by the optimizer.

        :type alpha: Variable
        :param alpha: a scale to be applied to the model parameter updates

        :rtype: iterable over pairs (shared variable, new expression)
        :returns: expressions how to update the optimizer parameters
        """

        result = []
        deltas = dict()
        for path, gradient in zip(self.network.get_variables(),
                                  self._gradients):
            ms_gradient_old = self._params[path + '_mean_sqr_gradient']
            ms_gradient = \
                self._gamma * ms_gradient_old + \
                (1.0 - self._gamma) * tensor.sqr(gradient)
            result.append((ms_gradient_old, ms_gradient))

            ms_velocity_old = self._params[path + '_mean_sqr_velocity']
            # rms_velocity quantity lags behind rms_gradient by 1 time step,
            # due to the recurrence relationship for velocity.
            rms_gradient = tensor.sqrt(ms_gradient + self._epsilon)
            rms_velocity = tensor.sqrt(ms_velocity_old + self._epsilon)
            velocity = -gradient * rms_velocity / rms_gradient
            deltas[path] = velocity
        self._normalize(deltas)

        for path, param_old in self.network.get_variables().items():
            delta = deltas[path]
            ms_velocity_old = self._params[path + '_mean_sqr_velocity']
            ms_velocity = self._gamma * ms_velocity_old + \
                          (1.0 - self._gamma) * tensor.sqr(delta)
            result.append((ms_velocity_old, ms_velocity))
            result.append((param_old, param_old + alpha * delta))
        return result
