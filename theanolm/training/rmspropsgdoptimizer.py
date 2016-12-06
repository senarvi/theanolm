#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import theano.tensor as tensor
from theanolm import Parameters
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
            self._params.add(path + '_gradient',
                             numpy.zeros_like(param.get_value()))
            # Initialize mean squared gradient to ones, otherwise the first
            # update will be divided by close to zero.
            self._params.add(path + '_mean_sqr_gradient',
                             numpy.ones_like(param.get_value()))

        # geometric rate for averaging gradients
        if not 'gradient_decay_rate' in optimization_options:
            raise ValueError("Gradient decay rate is not given in training "
                             "options.")
        self._gamma = optimization_options['gradient_decay_rate']

        super().__init__(optimization_options, network, *args, **kwargs)

    def _gradient_update_exprs(self):
        result = []
        for path, gradient_new in zip(self.network.get_variables(),
                                      self._gradient_exprs):
            gradient = self._params[path + '_gradient']
            ms_gradient = self._params[path + '_mean_sqr_gradient']
            ms_gradient_new = \
                self._gamma * ms_gradient + \
                (1.0 - self._gamma) * tensor.sqr(gradient_new)
            result.append((gradient, gradient_new))
            result.append((ms_gradient, ms_gradient_new))
        return result

    def _model_update_exprs(self, alpha):
        updates = dict()
        for path, param in self.network.get_variables().items():
            gradient = self._params[path + '_gradient']
            ms_gradient = self._params[path + '_mean_sqr_gradient']
            rms_gradient = tensor.sqrt(ms_gradient + self._epsilon)
            updates[path] = -gradient / rms_gradient
        self._normalize(updates)

        result = []
        for path, param in self.network.get_variables().items():
            update = updates[path]
            result.append((param, param + alpha * update))
        return result
