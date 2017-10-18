#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the RMSProp optimizer with SGD.
"""

import numpy
import theano.tensor as tensor

from theanolm.backend import Parameters
from theanolm.training.basicoptimizer import BasicOptimizer

class RMSPropSGDOptimizer(BasicOptimizer):
    """RMSProp Variation of Stochastic Gradient Descent Optimization Method

    At the time of writing, RMSProp is an unpublished method. Usually people
    cite slide 29 of Lecture 6 of Geoff Hinton's Coursera class:
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

    The idea is simply to maintain a running average of the squared gradient for
    each parameter, and divide the gradient by the root of the mean squared
    gradient (RMS). This makes RMSProp take steps near 1 whenever the gradient
    is of constant magnitude, and larger steps whenever the local scale of the
    gradient starts to increase.
    """

    def __init__(self, optimization_options, network, *args, **kwargs):
        """Creates an RMSProp SGD optimizer.

        :type optimization_options: dict
        :param optimization_options: a dictionary of optimization options

        :type network: Network
        :param network: the neural network object
        """

        self._params = Parameters()
        for path, param in network.get_variables().items():
            # Initialize mean squared gradient to ones, otherwise the first
            # update will be divided by close to zero.
            self._params.add(path + '_mean_sqr_gradient',
                             numpy.ones_like(param.get_value()))

        # geometric rate for averaging gradients
        if 'gradient_decay_rate' not in optimization_options:
            raise ValueError("Gradient decay rate is not given in training "
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

            rms_gradient = tensor.sqrt(ms_gradient + self._epsilon)
            deltas[path] = -gradient / rms_gradient
        self._normalize(deltas)

        for path, param_old in self.network.get_variables().items():
            delta = deltas[path]
            result.append((param_old, param_old + alpha * delta))
        return result
