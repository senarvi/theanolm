#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import time
import logging
import numpy
import theano
import theano.tensor as tensor
from theanolm.exceptions import IncompatibleStateError, NumberError

class ModelTrainer(object):
    """Superclass for Neural Network Language Model Trainers
    """

    def __init__(self, network, training_options, profile=False):
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

        :type network: Network
        :param network: the neural network object

        :type training_options: dict
        :param training_options: a dictionary of training options

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
        if 'max_gradient_norm' in training_options:
            max_norm = training_options['max_gradient_norm']
            epsilon = training_options['epsilon']
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

        For consistency, all the parameter values are returned as numpy types,
        since state read from a model file also contains numpy types.

        :rtype: dict of numpy types
        :returns: a dictionary of the parameter values
        """

        result = OrderedDict()
        for name, param in self.params.items():
            result[name] = param.get_value()
        result['epoch_number'] = numpy.int64(self.epoch_number)
        result['update_number'] = numpy.int64(self.update_number)
        result['cost_history'] = numpy.asarray(self._cost_history)
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

        if not 'cost_history' in state:
            raise IncompatibleStateError("Validation set cost history is "
                                         "missing from training state.")
        saved_cost_history = state['cost_history'].tolist()
        # If the error history was empty when the state was saved,
        # ndarray.tolist() will return None.
        if saved_cost_history is None:
            self._cost_history = []
        else:
            self._cost_history = saved_cost_history
        logging.debug("Validation set cost history since learning rate was "
                      "decreased:")
        logging.debug(str(numpy.asarray(self._cost_history)))

        if not 'epoch_number' in state:
            raise IncompatibleStateError("Current epoch number is missing from "
                                         "training state.")
        self.epoch_number = state['epoch_number'].item()
        if not 'update_number' in state:
            raise IncompatibleStateError("Current update number is missing "
                                         "from training state.")
        self.update_number = state['update_number'].item()
        logging.info("Restored training state from update %d.%d.",
            self.epoch_number, self.update_number)

    def update_minibatch(self, batch_iter):
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

        # Read the next mini-batch. StopIteration is risen at the end of input.
        try:
            word_ids, _, mask = next(batch_iter)
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
        self.model_update_function()

        # For log_update().
        self.update_duration = time.time() - update_start_time
        return True

    def log_update(self):
        """Logs information about the previous mini-batch update.
        """

        if 'trainer.learning_rate' in self.params:
            logging.info("Batch %d.%d (%d) -- lr = %g, cost = %.2f, duration = %.2f ms",
                    self.epoch_number,
                    self.update_number,
                    self.total_updates,
                    self.params['trainer.learning_rate'].get_value(),
                    self.update_cost,
                    self.update_duration * 100)
        else:
            logging.info("Batch %d.%d (%d) -- cost = %.2f, duration = %.2f ms",
                    self.epoch_number,
                    self.update_number,
                    self.total_updates,
                    self.update_cost,
                    self.update_duration * 100)

    def decrease_learning_rate(self):
        """Called when the validation set cost stops decreasing.
        """

        if 'trainer.learning_rate' in self.params:
            old_value = self.params['trainer.learning_rate'].get_value()
            new_value = old_value / 2
            self.params['trainer.learning_rate'].set_value(new_value)
            logging.info("Learning rate reduced from %g to %g." %
                (old_value, new_value))
        self._cost_history = []

    def reset(self):
        """Resets the optimizer timestep. May be called after decreasing
        learning rate, depending on the program options.
        """

        pass

    def append_validation_cost(self, validation_cost):
        """Adds the validation set cost to the cost history.

        :type validation_cost: float
        :param validation_cost: the new validation set cost to be added to the history
        """

        self._cost_history.append(validation_cost)
        logging.debug("Validation set cost history since learning rate was "
                      "decreased:")
        logging.debug(str(numpy.asarray(self._cost_history)))

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
