#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
from theanolm import Parameters
from theanolm.training.basicoptimizer import BasicOptimizer

class NesterovOptimizer(BasicOptimizer):
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

    def __init__(self, optimization_options, network, *args, **kwargs):
        """Creates a Nesterov momentum optimizer.

        :type optimization_options: dict
        :param optimization_options: a dictionary of optimization options

        :type network: Network
        :param network: the neural network object
        """

        self._params = Parameters()
        for path, param in network.get_variables().items():
            self._params.add(path + '_gradient',
                             numpy.zeros_like(param.get_value()))
            self._params.add(path + '_velocity',
                             numpy.zeros_like(param.get_value()))

        # momentum
        if not 'momentum' in optimization_options:
            raise ValueError("Momentum is not given in optimization options.")
        self._momentum = optimization_options['momentum']

        super().__init__(optimization_options, network, *args, **kwargs)

    def _gradient_update_exprs(self):
        result = []
        for path, gradient_new in zip(self.network.get_variables(),
                                      self._gradient_exprs):
            gradient = self._params[path + '_gradient']
            result.append((gradient, gradient_new))
        return result

    def _model_update_exprs(self, alpha):
        updates = dict()
        for path, param in self.network.get_variables().items():
            gradient = self._params[path + '_gradient']
            updates[path] = -gradient
        self._normalize(updates)

        result = []
        for path, param in self.network.get_variables().items():
            update = updates[path]
            velocity = self._params[path + '_velocity']
            velocity_new = self._momentum * velocity + alpha * update
            param_new = param + self._momentum * velocity_new + alpha * update
            result.append((velocity, velocity_new))
            result.append((param, param_new))
        return result
