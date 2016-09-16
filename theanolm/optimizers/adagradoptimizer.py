#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as tensor
from theanolm.optimizers.basicoptimizer import BasicOptimizer

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

        self.param_init_values = dict()
        for name, param in network.params.items():
            self.param_init_values[name + '_gradient'] = \
                numpy.zeros_like(param.get_value())
            self.param_init_values[name + '_sum_sqr_gradient'] = \
                numpy.zeros_like(param.get_value())

        super().__init__(optimization_options, network, *args, **kwargs)

    def _gradient_update_exprs(self):
        result = []
        for name, gradient_new in zip(self.network.params,
                                      self._gradient_exprs):
            gradient = self.params[name + '_gradient']
            ss_gradient = self.params[name + '_sum_sqr_gradient']
            ss_gradient_new = ss_gradient + tensor.sqr(gradient_new)
            result.append((gradient, gradient_new))
            result.append((ss_gradient, ss_gradient_new))
        return result

    def _model_update_exprs(self, alpha):
        updates = dict()
        for name, param in self.network.params.items():
            gradient = self.params[name + '_gradient']
            ss_gradient = self.params[name + '_sum_sqr_gradient']
            rss_gradient = tensor.sqrt(ss_gradient + self._epsilon)
            updates[name] = -gradient / rss_gradient
        self._normalize(updates)

        result = []
        for name, param in self.network.params.items():
            update = updates[name]
            result.append((param, param + alpha * update))
        return result
