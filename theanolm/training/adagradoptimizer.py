#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the AdaGrad optimizer.
"""

import numpy
import theano.tensor as tensor

from theanolm.backend import Parameters
from theanolm.training.basicoptimizer import BasicOptimizer

class AdaGradOptimizer(BasicOptimizer):
    """AdaGrad Optimization Method

    AdaGrad is a simple extension of Stochastic Gradient Descent that adapts the
    step size for each component, based on how frequently each component occurs
    in the gradients. At each update, the learning rate is divided by the root
    of the sum of squared gradients. (Actually, in this simpler form of the
    algorithm, the squared gradient is used to approximate the outer product of
    the gradient vector by itself.)

    J. Duchi, E. Hazan, Y. Singer (2011)
    Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
    Journal of Machine Learning Research 12: 2121-2159

    Note: When using a learning rate decreasing schedule, perhaps a running
    average of the historical gradients would be better than a sum.
    """

    def __init__(self, optimization_options, network, *args, **kwargs):
        """Creates an AdaGrad optimizer.

        :type optimization_options: dict
        :param optimization_options: a dictionary of optimization options

        :type network: Network
        :param network: the neural network object
        """

        self._params = Parameters()
        for path, param in network.get_variables().items():
            self._params.add(path + '_sum_sqr_gradient',
                             numpy.zeros_like(param.get_value()))

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
            ss_gradient_old = self._params[path + '_sum_sqr_gradient']
            ss_gradient = ss_gradient_old + tensor.sqr(gradient)
            result.append((ss_gradient_old, ss_gradient))

            rss_gradient = tensor.sqrt(ss_gradient + self._epsilon)
            deltas[path] = -gradient / rss_gradient
        self._normalize(deltas)

        for path, param_old in self.network.get_variables().items():
            delta = deltas[path]
            result.append((param_old, param_old + alpha * delta))
        return result
