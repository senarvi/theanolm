#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as tensor
from theanolm.optimizers.basicoptimizer import BasicOptimizer

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

        self.param_init_values = dict()
        for name, param in network.params.items():
            self.param_init_values[name + '_gradient'] = \
                numpy.zeros_like(param.get_value())
            # Initialize mean squared gradient to ones, otherwise the first
            # update will be divided by close to zero.
            self.param_init_values[name + '_mean_sqr_gradient'] = \
                numpy.ones_like(param.get_value())

        # geometric rate for averaging gradients
        if not 'gradient_decay_rate' in optimization_options:
            raise ValueError("Gradient decay rate is not given in training "
                             "options.")
        self._gamma = optimization_options['gradient_decay_rate']

        super().__init__(optimization_options, network, *args, **kwargs)

    def _gradient_update_exprs(self):
        result = []
        for name, gradient_new in zip(self.network.params,
                                      self._gradient_exprs):
            gradient = self.params[name + '_gradient']
            ms_gradient = self.params[name + '_mean_sqr_gradient']
            ms_gradient_new = \
                self._gamma * ms_gradient + \
                (1.0 - self._gamma) * tensor.sqr(gradient_new)
            result.append((gradient, gradient_new))
            result.append((ms_gradient, ms_gradient_new))
        return result

    def _model_update_exprs(self, alpha):
        updates = dict()
        for name, param in self.network.params.items():
            gradient = self.params[name + '_gradient']
            ms_gradient = self.params[name + '_mean_sqr_gradient']
            rms_gradient = tensor.sqrt(ms_gradient + self._epsilon)
            updates[name] = -gradient / rms_gradient
        self._normalize(updates)

        result = []
        for name, param in self.network.params.items():
            update = updates[name]
            result.append((param, param + alpha * update))
        return result
