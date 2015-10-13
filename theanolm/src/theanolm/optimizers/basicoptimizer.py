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

        # Calculate log probability of each word.
        logprobs = tensor.log(self.network.prediction_probs)
        # Set the log probability to 0 after sequence ends.
        logprobs = logprobs * self.network.minibatch_mask
        # Calculate the negative log probability normalized by the number of
        # training examples in the mini-batch. By taking a mean instead of a
        # sum, the gradients will also be normalized by the number of words.
        cost = -logprobs.mean()

        # Compute the symbolic expression for updating the gradient with regard
        # to each parameter.
        gradients = tensor.grad(cost, wrt=list(self.network.params.values()))

        # Normalize the norm of the gradients to given maximum value.
        if 'max_gradient_norm' in optimization_options:
            max_norm = optimization_options['max_gradient_norm']
            epsilon = optimization_options['epsilon']
            squares = [tensor.sqr(gradient) for gradient in gradients]
            sums = [tensor.sum(square) for square in squares]
            total_sum = sum(sums)  # sum over parameter variables
            norm = tensor.sqrt(total_sum)
            target_norm = tensor.clip(norm, 0.0, max_norm)
            gradients = [gradient * target_norm / (epsilon + norm)
                         for gradient in gradients]

        self._gradient_exprs = gradients
        self.gradient_update_function = \
            theano.function([self.network.minibatch_input, self.network.minibatch_mask],
                            cost,
                            updates=self._get_gradient_updates(),
                            profile=profile)

        self.model_update_function = \
            theano.function([],
                            [],
                            updates=self._get_model_updates(),
                            profile=profile)

    def _create_params(self):
        """Creates Theano shared variables from the initial parameter values.
        """

        self.params = {name: theano.shared(value, name)
                       for name, value in self.param_init_values.items()}

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
        if 'optimizer.learning_rate' in self.params:
            return self.params['optimizer.learning_rate'].get_value()
        else:
            return 0

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
            raise NumberError("Update {} cost computation resulted in a "
                              "numerical error.".format(self.update_number))
        self.model_update_function()
        self.update_duration = time.time() - update_start_time

    def log_update(self, updates_per_epoch):
        """Logs information about the previous mini-batch update.
        """

        if 'optimizer.learning_rate' in self.params:
            learning_rate = self.params['optimizer.learning_rate'].get_value()
        else:
            learning_rate = 0

        logging.info("Update %d (%.2f %%) of epoch %d -- "
                     "lr = %g, cost = %.2f, duration = %.2f ms",
                     self.update_number,
                     self.update_number / updates_per_epoch * 100,
                     self.epoch_number,
                     learning_rate,
                     self.update_cost,
                     self.update_duration * 100)

    def decrease_learning_rate(self):
        """Called when the validation set cost stops decreasing.
        """

        if 'optimizer.learning_rate' in self.params:
            old_value = self.params['optimizer.learning_rate'].get_value()
            new_value = old_value / 2
            self.params['optimizer.learning_rate'].set_value(new_value)
            logging.info("Learning rate reduced from %g to %g." %
                (old_value, new_value))

    def reset(self):
        """Resets the optimizer timestep. May be called after decreasing
        learning rate, depending on the program options.
        """

        pass
