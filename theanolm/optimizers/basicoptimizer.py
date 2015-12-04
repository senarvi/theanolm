#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import time
import logging
import numpy
import theano
import theano.tensor as tensor
from theanolm.exceptions import IncompatibleStateError, NumberError

class BasicOptimizer(object):
    """Superclass for Neural Network Language Model Optimizers
    """

    def __init__(self, optimization_options, network, profile=False):
        """Creates Theano functions for training a neural network language
        model.

        The subclass constructor is expected to give default values to all the
        required parameters in self.param_init_values first. This constructor
        will then create the corresponding Theano shared variables, and two
        update functions:

        * gradient_update_function: updates the gradient parameters and returns
          the cost
        * model_update_function: updates model state given the gradients and the
          learning rate

        :type optimization_options: dict
        :param optimization_options: a dictionary of optimization options

        :type network: Network
        :param network: the neural network object

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        self.network = network

        # Create Theano shared variables from the initial parameter values.
        self.params = {name: theano.shared(value, name)
                       for name, value in self.param_init_values.items()}

        # numerical stability / smoothing term to prevent divide-by-zero
        if not 'epsilon' in optimization_options:
            raise ValueError("Epsilon is not given in optimization options.")
        self._epsilon = optimization_options['epsilon']

        # maximum norm for parameter updates
        if 'max_gradient_norm' in optimization_options:
            self._max_gradient_norm = optimization_options['max_gradient_norm']
        else:
            self._max_gradient_norm = None

        # Derive the symbolic expression for log probability of each word.
        logprobs = tensor.log(self.network.prediction_probs)
        # Set the log probability to 0, if the next input word (the one
        # predicted) is masked out.
        logprobs = logprobs * self.network.minibatch_mask[1:]
        # Cost is the negative log probability normalized by the number of
        # training examples in the mini-batch, so that the gradients will also
        # be normalized by the number of training examples.
        cost = -logprobs.sum() / self.network.minibatch_mask[1:].sum()

        # Derive the symbolic expression for updating the gradient with regard
        # to each parameter.
        self._gradient_exprs = \
            tensor.grad(cost, wrt=list(self.network.params.values()))

        self.gradient_update_function = \
            theano.function([self.network.minibatch_input,
                             self.network.minibatch_mask],
                            cost,
                            updates=self._get_gradient_updates(),
                            profile=profile)

        self.model_update_function = \
            theano.function([],
                            [],
                            updates=self._get_model_updates(),
                            profile=profile)

    def get_state(self):
        """Pulls parameter values from Theano shared variables.

        For consistency, all the parameter values are returned as numpy types,
        since state read from a model file also contains numpy types.

        :rtype: dict of numpy types
        :returns: a dictionary of the parameter values
        """

        result = OrderedDict()
        for name, param in self.params.items():
            result[name] = param.get_value()
        return result

    def set_state(self, state):
        """Sets the values of Theano shared variables.
        
        Requires that ``state`` contains values for all the training parameters.

        :type state: dict of numpy types
        :param state: a dictionary of training parameters
        """

        for name, param in self.params.items():
            if not name in state:
                raise IncompatibleStateError("Parameter %s is missing from "
                                             "training state." % name)
            new_value = state[name]
            param.set_value(new_value)
            if len(new_value.shape) == 0:
                logging.debug("%s <- %s", name, str(new_value))
            else:
                logging.debug("%s <- array%s", name, str(new_value.shape))

    def get_learning_rate(self):
        """Returns the current value of the learning rate.

        :rtype: float
        :returns: current learning rate, or 1.0 if not used by this optimization
                  method
        """

        if 'optimizer.learning_rate' in self.params:
            return self.params['optimizer.learning_rate'].get_value()
        else:
            return 1.0

    def set_learning_rate(self, x):
        """Sets a new value for the learning rate, if it is used by this
        optimization method.

        :type x: float
        :param x: new value for learning rate
        """

        if 'optimizer.learning_rate' in self.params:
            self.params['optimizer.learning_rate'].set_value(x)

    def update_minibatch(self, word_ids, mask):
        """Optimizes the neural network parameters using the given inputs and
        learning rate.

        ``batch_iter`` is an iterator to the training data. On each call it
        creates a tuple of three 2-dimensional matrices, all indexed by time
        step and sequence. The first matrix contains the word IDs, the second
        one (class membership probabilities) will be ignored in training, and
        the third one masks out elements past the sequence ends.

        :type batch_iter: BatchIterator
        :param batch_iter: an iterator that creates mini-batches from the
                           training data

        :rtype: bool
        :returns: True if an update was performed, False if there was no more
                  training data
        """

        update_start_time = time.time()
        self.update_cost = self.gradient_update_function(word_ids, mask)
        if numpy.isnan(self.update_cost) or numpy.isinf(self.update_cost):
            raise NumberError("Mini-batch cost computation resulted in a "
                              "numerical error.")
        self.model_update_function()
        self.update_duration = time.time() - update_start_time

    def reset(self):
        """Resets the optimizer timestep. May be called after decreasing
        learning rate, depending on the program options.
        """

        pass

    def _normalize(self, updates):
        """Normalizes the norm of a parameter update to given maximum value.

        :type updates: dict of str to theano.tensor.var.TensorVariable
        :param updates: dictionary of symbolic variables that describe the
                        negative gradient of each parameter, after any
                        optimization method specific adaptation

        :rtype: dict of str to theano.tensor.var.TensorVariable
        :returns: dictionary of symbolic variables that describe ``updates``
                  after normalization has been applied
        """

        max_norm = self._max_gradient_norm
        if max_norm is None:
            return

        squares = [tensor.sqr(update) for update in updates.values()]
        sums = [tensor.sum(square) for square in squares]
        total_sum = sum(sums)  # sum over parameter variables
        norm = tensor.sqrt(total_sum)
        target_norm = tensor.clip(norm, 0.0, max_norm)
        for name, update in updates.items():
            updates[name] = update * target_norm / (self._epsilon + norm)
