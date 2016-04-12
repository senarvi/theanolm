#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
from time import time
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

        # class IDs to ignore when computing the cost
        if 'classes_to_ignore' in optimization_options:
            classes_to_ignore = optimization_options['classes_to_ignore']
        else:
            classes_to_ignore = []

        # Derive the symbolic expression for log probability of each word.
        logprobs = tensor.log(self.network.prediction_probs)
        # Set the log probability to 0, if the next input word (the one
        # predicted) is masked out or to be ignored. The mask has to be cast to
        # floatX, otherwise the result will be float64 and pulled out from the
        # GPU earlier than necessary.
        mask = self.network.mask[1:]
        for class_id in classes_to_ignore:
            mask *= tensor.neq(self.network.input[1:], class_id)
        logprobs *= tensor.cast(mask, theano.config.floatX)
        # Cost is the negative log probability normalized by the number of
        # training examples in the mini-batch, so that the gradients will also
        # be normalized by the number of training examples.
        cost = -logprobs.sum() / tensor.cast(mask.sum(), theano.config.floatX)

        # Derive the symbolic expression for updating the gradient with regard
        # to each parameter.
        self._gradient_exprs = \
            tensor.grad(cost, wrt=list(self.network.params.values()))

        # Ignore unused input, because is_training is only used by dropout
        # layer.
        self.gradient_update_function = theano.function(
            [self.network.word_input,
             self.network.class_input,
             self.network.mask],
            cost,
            givens=[(self.network.is_training, numpy.int8(1))],
            updates=self._get_gradient_updates(),
            name='gradient_updater',
            on_unused_input='ignore',
            profile=profile)

        self.model_update_function = theano.function(
            [],
            [],
            updates=self._get_model_updates(),
            name='model_updater',
            profile=profile)

    def get_state(self, state):
        """Pulls parameter values from Theano shared variables.

        If there already is a parameter in the state, it will be replaced, so it
        has to have the same number of elements.

        :type state: h5py.File
        :param state: HDF5 file for storing the optimization parameters
        """

        for name, param in self.params.items():
            if name in state:
                state[name][:] = param.get_value()
            else:
                state.create_dataset(name, data=param.get_value())

    def set_state(self, state):
        """Sets the values of Theano shared variables.
        
        Requires that ``state`` contains values for all the optimization
        parameters.

        :type state: h5py.File
        :param state: HDF5 file that contains the optimization parameters
        """

        for name, param in self.params.items():
            if not name in state:
                raise IncompatibleStateError("Parameter %s is missing from "
                                             "training state." % name)
            new_value = state[name].value
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

        if 'optimizer/learning_rate' in self.params:
            return self.params['optimizer/learning_rate'].get_value()
        else:
            return 1.0

    def set_learning_rate(self, x):
        """Sets a new value for the learning rate, if it is used by this
        optimization method.

        :type x: float
        :param x: new value for learning rate
        """

        if 'optimizer/learning_rate' in self.params:
            self.params['optimizer/learning_rate'].set_value(x)

    def update_minibatch(self, word_ids, class_ids, mask):
        """Optimizes the neural network parameters using the given inputs and
        learning rate.

        :type word_ids: numpy.ndarray of an integer type
        :param word_ids: a 2-dimensional matrix, indexed by time step and
                         sequence, that contains the word IDs

        :type class_ids: numpy.ndarray of an integer type
        :param class_ids: a 2-dimensional matrix, indexed by time step and
                          sequence, that contains the class IDs

        :type mask: numpy.ndarray of a floating point type
        :param mask: a 2-dimensional matrix, indexed by time step and sequence,
                     that masks out elements past the sequence ends.
        """

        update_start_time = time()
        self.update_cost = self.gradient_update_function(word_ids, class_ids,
                                                         mask)
        if numpy.isnan(self.update_cost) or numpy.isinf(self.update_cost):
            raise NumberError("Mini-batch cost computation resulted in a "
                              "numerical error.")
        self.model_update_function()
        self.update_duration = time() - update_start_time

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
