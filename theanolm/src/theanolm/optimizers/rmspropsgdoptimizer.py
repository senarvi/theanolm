#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as tensor
from theanolm.optimizers.basicoptimizer import BasicOptimizer

class RMSPropSGDOptimizer(BasicOptimizer):
    """RMSProp Variation of Stochastic Gradient Descent Optimization Method
    
    RMSProp is currently an unpublished method. Usually people cite slide 29
    of Lecture 6 of Geoff Hinton's Coursera class:
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf 
    
    The idea is simply to maintain a running average of the squared gradient for
    each parameter, and divide the gradient by the root of the mean squared
    gradient (RMS). This makes RMSProp take steps near 1 whenever the gradient
    is of constant magnitude, and larger steps whenever the local scale of the
    gradient starts to increase.
    """

    def __init__(self, network, optimization_options, profile):
        """Creates an RMSProp optimizer.

        :type network: Network
        :param network: the neural network object

        :type optimization_options: dict
        :param optimization_options: a dictionary of optimization options

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
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
            self.param_init_values[name + '.gradient'] = \
                numpy.zeros_like(param.get_value())
            self.param_init_values[name + '.mean_sqr_gradient'] = \
                numpy.zeros_like(param.get_value())

        self._create_params()

        # geometric rate for averaging gradients
        if not 'gradient_decay_rate' in optimization_options:
            raise ValueError("Gradient decay rate is not given in training "
                             "options.")
        self._gamma = optimization_options['gradient_decay_rate']

        # numerical stability / smoothing term to prevent divide-by-zero
        if not 'epsilon' in optimization_options:
            raise ValueError("Epsilon is not given in optimization options.")
        self._epsilon = optimization_options['epsilon']

        super().__init__(network, optimization_options, profile)

    def _get_gradient_updates(self):
        result = []
        for name, gradient_new in zip(self.network.params, self._gradient_exprs):
            gradient = self.params[name + '.gradient']
            ms_gradient = self.params[name + '.mean_sqr_gradient']
            ms_gradient_new = (self._gamma * ms_gradient) + ((1.0 - self._gamma) * tensor.sqr(gradient_new))
            result.append((gradient, gradient_new))
            result.append((ms_gradient, ms_gradient_new))
        return result

    def _get_model_updates(self):
        alpha = self.params['optimizer.learning_rate']

        result = []
        for name, param in self.network.params.items():
            gradient = self.params[name + '.gradient']
            ms_gradient = self.params[name + '.mean_sqr_gradient']
            rms_gradient = tensor.sqrt(ms_gradient + self._epsilon)
            param_new = param - (alpha * gradient / rms_gradient)
            result.append((param, param_new))
        return result
