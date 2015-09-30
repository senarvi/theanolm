#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from theanolm.trainers.modeltrainer import ModelTrainer
import theano.tensor as tensor
import numpy

class AdadeltaTrainer(ModelTrainer):
    """Adadelta Optimization Method

    Zeiler, M. D.
    ADADELTA: An adaptive learning rate method
    CoRR, 2012
    http://arxiv.org/abs/1212.5701
    """

    def __init__(self, network, profile):
        """Creates an Adadelta trainer.

        :type network: Network
        :param network: the neural network object

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        self.param_init_values = dict()
        for name, param in network.params.items():
            self.param_init_values[name + '.gradient'] = numpy.zeros_like(param.get_value())
            self.param_init_values[name + '.mean_sqr_gradient'] = numpy.zeros_like(param.get_value())
            self.param_init_values[name + '.mean_sqr_velocity'] = numpy.zeros_like(param.get_value())
        self._create_params()
        self._gamma = 0.95  # geometric rate for averaging gradients (decay rate)
        self._epsilon = 1e-6

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
            ms_velocity = self.params[name + '.mean_sqr_velocity']
            # rms_velocity quantity lags behind rms_gradient by 1 time step,
            # due to the recurrence relationship for velocity.
            rms_gradient = tensor.sqrt(ms_gradient + self._epsilon)
            rms_velocity = tensor.sqrt(ms_velocity + self._epsilon)
            velocity = -(rms_velocity / rms_gradient) * gradient
            ms_velocity_new = (self._gamma * ms_velocity) + ((1.0 - self._gamma) * tensor.sqr(velocity))
            param_new = param + velocity
            result.append((ms_velocity, ms_velocity_new))
            result.append((param, param_new))
        return result
