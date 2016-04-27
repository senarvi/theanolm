#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as tensor
from theanolm.optimizers.basicoptimizer import BasicOptimizer

class AdadeltaOptimizer(BasicOptimizer):
    """ADADELTA Optimization Method

    ADADELTA optimization method has been derived from AdaGrad. AdaGrad
    accumulates the sum of squared gradients over all time, which is used to
    scale the learning rate smaller and smaller. ADADELTA uses an exponentially
    decaying average of the squared gradients.

    This implementation scales the parameter updates by the learning rate
    hyperparameter. The original paper does not include such scaling,
    corresponding to learning rate 1.

    M. D. Zeiler (2012)
    ADADELTA: An adaptive learning rate method
    http://arxiv.org/abs/1212.5701
    """

    def __init__(self, optimization_options, network, *args, **kwargs):
        """Creates an Adadelta optimizer.

        :type optimization_options: dict
        :param optimization_options: a dictionary of optimization options

        :type network: Network
        :param network: the neural network object
        """

        self.param_init_values = dict()
        for name, param in network.params.items():
            self.param_init_values[name + '_gradient'] = \
                numpy.zeros_like(param.get_value())
            self.param_init_values[name + '_mean_sqr_gradient'] = \
                numpy.zeros_like(param.get_value())
            self.param_init_values[name + '_mean_sqr_velocity'] = \
                numpy.zeros_like(param.get_value())

        # geometric rate for averaging gradients
        if not 'gradient_decay_rate' in optimization_options:
            raise ValueError("Gradient decay rate is not given in optimization "
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
            ms_velocity = self.params[name + '_mean_sqr_velocity']
            # rms_velocity quantity lags behind rms_gradient by 1 time step,
            # due to the recurrence relationship for velocity.
            rms_gradient = tensor.sqrt(ms_gradient + self._epsilon)
            rms_velocity = tensor.sqrt(ms_velocity + self._epsilon)
            velocity = -gradient * rms_velocity / rms_gradient
            updates[name] = velocity
        self._normalize(updates)

        result = []
        for name, param in self.network.params.items():
            update = updates[name]
            ms_velocity = self.params[name + '_mean_sqr_velocity']
            ms_velocity_new = self._gamma * ms_velocity + \
                              (1.0 - self._gamma) * tensor.sqr(update)
            param_new = param + alpha * update
            result.append((ms_velocity, ms_velocity_new))
            result.append((param, param_new))
        return result
