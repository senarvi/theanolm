#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import time
import theano
import theano.tensor as tensor
import numpy
from theanolm.exceptions import IncompatibleStateError, NumberError

class ModelTrainer(object):
    """Superclass for Neural Network Language Model Trainers
    """

    def __init__(self, network, profile=False):
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

        :type network: RNNLM
        :param network: the neural network object

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        self.network = network

        # Calculate negative log probability of each word.
        costs = -tensor.log(self.network.training_probs)
        # Apply mask to the costs matrix.
        costs = costs * self.network.minibatch_mask
        # Sum costs over time steps and take the average over sequences.
        cost = costs.sum(0).mean()
        # Compute the symbolic rule for updating the gradient of each parameter.
        self._gradient_wrt_params = \
                tensor.grad(cost, wrt=list(self.network.params.values()))
        self.gradient_update_function = \
            theano.function([self.network.minibatch_input, self.network.minibatch_mask],
                            cost,
                            updates=self._get_gradient_updates(),
                            profile=profile)

        self.learning_rate = tensor.scalar('learning_rate', dtype='float32')
        self.learning_rate.tag.test_value = numpy.float32(0.001)
        self.model_update_function = \
            theano.function([self.learning_rate],
                            [],
                            updates=self._get_model_updates(),
                            on_unused_input='ignore',
                            profile=profile)

        # current training epoch
        self.epoch_number = 1
        # number of mini-batch updates performed in this epoch
        self.update_number = 0
        # total number of mini-batch updates performed (after restart)
        self.total_updates = 0
        # validation set cost history
        self._cost_history = []

    def _create_params(self):
        """Creates Theano shared variables from the initial parameter values.
        """

        self.params = {name: theano.shared(value, name)
                       for name, value in self.param_init_values.items()}

    def get_state(self):
        """Pulls parameter values from Theano shared variables.

        :rtype: dict
        :returns: a dictionary of the parameter values
        """

        result = OrderedDict()
        for name, param in self.params.items():
            result[name] = param.get_value()
        result['epoch_number'] = self.epoch_number
        result['update_number'] = self.update_number
        result['cost_history'] = self._cost_history
        return result

    def set_state(self, state):
        """Sets the values of Theano shared variables.
        
        Requires that ``state`` contains values for all the training parameters.

        :type state: dict
        :param state: dictionary of training parameters
        """

        for name, param in self.params.items():
            if not name in state:
                raise IncompatibleStateError("Parameter %s is missing from training state." % name)
            param.set_value(state[name])

        if not 'cost_history' in state:
            raise IncompatibleStateError("Validation set cost history is missing from training state.")
        saved_cost_history = state['cost_history'].tolist()
        # If the error history was empty when the state was saved,
        # ndarray.tolist() will return None.
        if saved_cost_history is None:
            self._cost_history = []
        else:
            self._cost_history = saved_cost_history
        self.print_cost_history()

        if not 'epoch_number' in state:
            raise IncompatibleStateError("Current epoch number is missing from training state.")
        self.epoch_number = state['epoch_number']
        if not 'update_number' in state:
            raise IncompatibleStateError("Current update number is missing from training state.")
        self.update_number = state['update_number']
        print("Previous training was stopped after update %d.%d." % (self.epoch_number, self.update_number))

    def print_cost_history(self):
        """Prints the current cost history.
        
        We're using numpy for prettier formatting.
        """

        print("Validation set cost history since last reset:")
        print(numpy.asarray(self._cost_history))

    def update_minibatch(self, batch_iter, learning_rate):
        """Optimizes the neural network parameters using the given inputs
        and learning rate.

        ``batch_iter`` is an iterator to the training data. On each call
        it creates a tuple of two 2-dimensional matrices, both indexed by
        time step and sequence. The first matrix contains the word IDs,
        and the second one contains the mask.

        :type batch_iter: BatchIterator
        :param batch_iter: an iterator creating mini-batches from the
                           training data

        :type mask: numpy.matrixlib.defmatrix.matrix
        :param mask: a matrix the same shape as x that contains 0.0 where x
                     contains input and 1.0 after sequence end

        :type learning_rate: float
        :param learning_rate: learning rate for the optimization

        :rtype: bool
        :returns: True if an update was performed, False if there was no more
                  training data
        """

        # Read the next mini-batch. StopIterator is risen at the end of input.
        try:
            word_ids, mask = next(batch_iter)
        except StopIteration:
            self.epoch_number += 1
            self.update_number = 0
            return False

        self.update_number += 1
        self.total_updates += 1

        update_start_time = time.time()
        self.update_cost = self.gradient_update_function(word_ids, mask)
        if numpy.isnan(self.update_cost):
            raise NumberError("Update %d cost has NaN value." % self.update_number)
        if numpy.isinf(self.update_cost):
            raise NumberError("Update %d cost has infinite value." % self.update_number)
        self.model_update_function(learning_rate)

        # For print_update_states().
        self.update_duration = time.time() - update_start_time
        self.update_learning_rate = learning_rate
        return True

    def print_update_stats(self):
        """Print information about the previous mini-batch update.
        """

        print("Batch %d.%d (%d) -- lr = %f, cost = %f, duration = %f seconds" % (\
                self.epoch_number,
                self.update_number,
                self.total_updates,
                self.update_learning_rate,
                self.update_cost,
                self.update_duration))

    def next_epoch(self):
        """Skips to the next epoch.

        Called when the validation set cost stops decreasing.
        """

        self.epoch_number += 1
        self.update_number = 0
        self._cost_history = []
        self.reset()

    def reset(self):
        """Resets training parameters if necessary before starting the next epoch
        after cost won't decrease.
        """
        pass

    def append_validation_cost(self, validation_cost):
        """Adds the validation set cost to the cost history.

        :type validation_cost: float
        :param validation_cost: the new validation set cost to be added to the history
        """

        self._cost_history.append(validation_cost)

    def validations_since_min_cost(self):
        """Returns the number of times the validation set cost has been computed
        since the minimum cost was obtained.

        :rtype: int
        :returns: number of validations since the minimum cost (0 means the last
                  validation is the best so far)
        """

        if len(self._cost_history) == 0:
            raise RuntimeError("ModelTrainer.validations_since_min_cost() called with empty cost history.")
        else:
            # Reverse the order of self._cost_history to find the last element
            # with the minimum value (in case there are several elements with the
            # same value.
            return numpy.argmin(self._cost_history[::-1])
