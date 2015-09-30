#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from theanolm.trainers.modeltrainer import ModelTrainer
import theano.tensor as tensor
import numpy

class RMSPropMomentumTrainer(ModelTrainer):
    """RMSProp variation of Momentum Optimization Method
    
    RMSProp is currently an unpublished method. Usually people cite slide 29
    of Lecture 6 of Geoff Hinton's Coursera class:
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf 
    
    The idea is simply to maintain a running average of the squared gradient for
    each parameter, and divide the gradient by the root of the mean squared
    gradient (RMS). This makes RMSProp take steps near 1 whenever the gradient
    is of constant magnitude, and larger steps whenever the local scale of the
    gradient starts to increase.

    RMSProp has been implemented over many optimization methods. This algorithm
    is from:

    A. Graves
    Generating Sequences With Recurrent Neural Networks
    Corr, 2013
    http://arxiv.org/abs/1308.0850
    """

    def __init__(self, network, training_options, profile=False):
        """Creates an RMSProp trainer.

        :type network: Network
        :param network: the neural network object

        :type training_options: dict
        :param training_options: a dictionary of training options

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        self.param_init_values = dict()
        for name, param in network.params.items():
            self.param_init_values[name + '.gradient'] = numpy.zeros_like(param.get_value())
            self.param_init_values[name + '.mean_gradient'] = numpy.zeros_like(param.get_value())
            self.param_init_values[name + '.mean_sqr_gradient'] = numpy.zeros_like(param.get_value())
            self.param_init_values[name + '.velocity'] = numpy.zeros_like(param.get_value())
        self._create_params()

        # geometric rate for averaging gradients
        if not 'gradient_decay_rate' in training_options:
            raise ValueError("Gradient decay rate is not given in training options.")
        self._gamma = training_options['gradient_decay_rate']

        # numerical stability / smoothing term to prevent divide-by-zero
        if not 'epsilon' in training_options:
            raise ValueError("Epsilon is not given in training options.")
        self._epsilon = training_options['epsilon']

        # momentum
        if not 'momentum' in training_options:
            raise ValueError("Momentum is not given in training options.")
        self._momentum = training_options['momentum']

        super().__init__(network, training_options, profile)

    def _get_gradient_updates(self):
        result = []
        for name, gradient_new in zip(self.network.params, self._gradient_wrt_params):
            gradient = self.params[name + '.gradient']
            m_gradient = self.params[name + '.mean_gradient']
            ms_gradient = self.params[name + '.mean_sqr_gradient']
            m_gradient_new = (self._gamma * m_gradient) + ((1.0 - self._gamma) * gradient_new)
            ms_gradient_new = (self._gamma * ms_gradient) + ((1.0 - self._gamma) * tensor.sqr(gradient_new))
            result.append((gradient, gradient_new))
            result.append((m_gradient, m_gradient_new))
            result.append((ms_gradient, ms_gradient_new))
        return result

    def _get_model_updates(self):
        result = []
        for name, param in self.network.params.items():
            gradient = self.params[name + '.gradient']
            m_gradient = self.params[name + '.mean_gradient']
            ms_gradient = self.params[name + '.mean_sqr_gradient']
            velocity = self.params[name + '.velocity']
            # I don't know why the square of average gradient is subtracted, but
            # I've seen this used when RMSProp is implemented with a momentum
            # method.
#            rms_gradient = tensor.sqrt(ms_gradient + self._epsilon)
            rms_gradient = tensor.sqrt(ms_gradient - tensor.sqr(m_gradient) + self._epsilon)
            velocity_new = (self._momentum * velocity) - (self.learning_rate * gradient / rms_gradient)
            param_new = param + velocity_new
            result.append((velocity, velocity_new))
            result.append((param, param_new))
        return result
