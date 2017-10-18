#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the Nesterov momentum optimizer.
"""

import numpy

from theanolm.backend import Parameters
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
        """Creates a Nesterov momentum optimizer. Nesterov momentum optimizer
        does not use additional parameters.

        :type optimization_options: dict
        :param optimization_options: a dictionary of optimization options

        :type network: Network
        :param network: the neural network object
        """

        self._params = Parameters()
        for path, param in network.get_variables().items():
            self._params.add(path + '_velocity',
                             numpy.zeros_like(param.get_value()))

        # momentum
        if 'momentum' not in optimization_options:
            raise ValueError("Momentum is not given in optimization options.")
        self._momentum = optimization_options['momentum']

        super().__init__(optimization_options, network, *args, **kwargs)

    def _get_param_updates(self, alpha):
        """Returns Theano expressions for updating the model parameters and any
        additional parameters required by the optimizer.

        :type alpha: Variable
        :param alpha: a scale to be applied to the model parameter updates

        :rtype: iterable over pairs (shared variable, new expression)
        :returns: expressions how to update the optimizer parameters
        """

        result = []
        deltas = dict()
        for path, gradient in zip(self.network.get_variables(),
                                  self._gradients):
            deltas[path] = -gradient
        self._normalize(deltas)

        for path, param_old in self.network.get_variables().items():
            delta = deltas[path]
            velocity_old = self._params[path + '_velocity']
            velocity = self._momentum * velocity_old + alpha * delta
            param = param_old + self._momentum * velocity + alpha * delta
            result.append((velocity_old, velocity))
            result.append((param_old, param))
        return result
