#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
from theanolm.trainers.modeltrainer import ModelTrainer

class NesterovTrainer(ModelTrainer):
    """Nesterov Momentum Optimization Method

    Normally Nesterov momentum is implemented by first taking a step towards
    the previous update direction, calculating gradient at that position,
    using the gradient to obtain the new update direction, and finally
    updating the parameters. We use an alternative formulation that requires
    the gradient to be computed only at the current parameter values,
    described here:
    https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617

    v_{t} = mu * v_{t-1} - lr * gradient(params_{t-1})
    params_{t} = params_{t-1} + mu * v_{t} - lr * gradient(params_{t-1})
    """

    def __init__(self, network, training_options, profile=False):
        """Creates a Nesterov momentum trainer.

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
            self.param_init_values[name + '.velocity'] = numpy.zeros_like(param.get_value())
        self._create_params()

        # momentum
        if not 'momentum' in training_options:
            raise ValueError("Momentum is not given in training options.")
        self._momentum = training_options['momentum']

        super().__init__(network, training_options, profile)

    def _get_gradient_updates(self):
        result = []
        for name, gradient_new in zip(self.network.params, self._gradient_exprs):
            gradient = self.params[name + '.gradient']
            result.append((gradient, gradient_new))
        return result

    def _get_model_updates(self):
        result = []
        for name, param in self.network.params.items():
            gradient = self.params[name + '.gradient']
            velocity = self.params[name + '.velocity']
            standard_update = -(self.learning_rate * gradient)
            velocity_new = (self._momentum * velocity) + standard_update
            param_new = param + (self._momentum * velocity_new) + standard_update
            result.append((velocity, velocity_new))
            result.append((param, param_new))
        return result
