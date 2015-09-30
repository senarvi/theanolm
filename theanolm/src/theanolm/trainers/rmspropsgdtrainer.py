#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from theanolm.trainers.modeltrainer import ModelTrainer
import theano.tensor as tensor
import numpy

class RMSPropSGDTrainer(ModelTrainer):
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

    def __init__(self, network, profile):
        """Creates an RMSProp trainer.

        :type network: Network
        :param network: the neural network object

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        self.param_init_values = dict()
        for name, param in network.params.items():
            self.param_init_values[name + '.gradient'] = numpy.zeros_like(param.get_value())
            self.param_init_values[name + '.mean_sqr_gradient'] = numpy.zeros_like(param.get_value())
        self._create_params()
        self._gamma = 0.95  # geometric rate for averaging gradients (decay rate)
        self._epsilon = 1e-6  # numerical stability / smoothing term to prevent divide-by-zero

        super().__init__(network, profile)

    def _get_gradient_updates(self):
        result = []
        for name, gradient_new in zip(self.network.params, self._gradient_wrt_params):
            gradient = self.params[name + '.gradient']
            ms_gradient = self.params[name + '.mean_sqr_gradient']
            ms_gradient_new = (self._gamma * ms_gradient) + ((1.0 - self._gamma) * tensor.sqr(gradient_new))
            result.append((gradient, gradient_new))
            result.append((ms_gradient, ms_gradient_new))
        return result

    def _get_model_updates(self):
        result = []
        for name, param in self.network.params.items():
            gradient = self.params[name + '.gradient']
            ms_gradient = self.params[name + '.mean_sqr_gradient']
            rms_gradient = tensor.sqrt(ms_gradient + self._epsilon)
            param_new = param - (self.learning_rate * gradient / rms_gradient)
            result.append((param, param_new))
        return result
