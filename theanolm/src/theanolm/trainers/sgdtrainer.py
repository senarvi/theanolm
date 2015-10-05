#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import theano
from theanolm.trainers.modeltrainer import ModelTrainer

class SGDTrainer(ModelTrainer):
    """Stochastic Gradient Descent Optimization Method
    """

    def __init__(self, network, training_options, profile):
        """Creates a Stochastic Gradient Descent trainer.

        :type network: Network
        :param network: the neural network object

        :type training_options: dict
        :param training_options: a dictionary of training options

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        self.param_init_values = dict()

        # Learning rate / step size will change during the iterations, so we'll
        # make it a shared variable.
        if not 'learning_rate' in training_options:
            raise ValueError("Learning rate is not given in training options.")
        self.param_init_values['trainer.learning_rate'] = \
            numpy.dtype(theano.config.floatX).type(
                training_options['learning_rate'])

        for name, param in network.params.items():
            self.param_init_values[name + '.gradient'] = numpy.zeros_like(param.get_value())

        self._create_params()

        super().__init__(network, training_options, profile)

    def _get_gradient_updates(self):
        result = []
        for name, gradient_new in zip(self.network.params, self._gradient_exprs):
            gradient = self.params[name + '.gradient']
            result.append((gradient, gradient_new))
        return result

    def _get_model_updates(self):
        alpha = self.params['trainer.learning_rate']

        result = []
        for name, param in self.network.params.items():
            gradient = self.params[name + '.gradient']
            result.append((param, param - alpha * gradient))
        return result
