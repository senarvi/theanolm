#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from theanolm.trainers.modeltrainer import ModelTrainer
import numpy

class SGDTrainer(ModelTrainer):
    """Stochastic Gradient Descent Optimization Method
    """

    def __init__(self, network, profile):
        """Creates a Stochastic Gradient Descent trainer.

        :type network: Network
        :param network: the neural network object

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        self.param_init_values = \
            {name + '.gradient': numpy.zeros_like(param.get_value())
             for name, param in network.params.items()}
        self._create_params()

        super().__init__(network, profile)

    def _get_gradient_updates(self):
        result = []
        for name, gradient_new in zip(self.network.params, self._gradient_wrt_params):
            gradient = self.params[name + '.gradient']
            result.append((gradient, gradient_new))
        return result

    def _get_model_updates(self):
        result = []
        for name, param in self.network.params.items():
            gradient = self.params[name + '.gradient']
            result.append((param, param - self.learning_rate * gradient))
        return result
