#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the BasicOptimizer class, a base class for
optimizers.
"""

from abc import abstractmethod, ABCMeta

import numpy
import theano
import theano.tensor as tensor

from theanolm.backend import IncompatibleStateError
from theanolm.backend import test_value
from theanolm.backend import sum_of_squares

class BasicOptimizer(object, metaclass=ABCMeta):
    """Superclass for Neural Network Language Model Optimizers
    """

    def __init__(self, optimization_options, network, cost_function,
                 profile=False):
        """Creates Theano functions for training a neural network language
        model.

        The subclass constructor is expected to create the optimizer parameters
        in ``self._params``. This constructor will then create a function
        ``self.update_function``, which updates the optimizer parameters, and
        then the model state given the gradients, the optimizer parameters, and the
        learning rate.

        The update functions takes as arguments four matrices and the alpha
        hyperparameter:

        1. Word IDs in the shape of a mini-batch. The functions will slice this
           into input and output.
        2. Class IDs in the shape of a mini-batch. The functions will slice this
           into input and output.
        3. Mask in the shape of a mini-batch, but only for the output words (not
           for the first time step).
        4. Weights in the shape of a mini-batch, but only for the output words
           (not for the first time step).
        4. Alpha or learning rate is used to scale the size of the update.

        :type optimization_options: dict
        :param optimization_options: a dictionary of optimization options

        :type network: Network
        :param network: the neural network object

        :type cost_function: Cost
        :param cost_function: an object from one of the cost function classes
                              that defined the training objective

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        self.network = network

        float_type = numpy.dtype(theano.config.floatX).type
        self.float_type = float_type

        try:
            # numerical stability / smoothing term to prevent divide-by-zero
            self._epsilon = float_type(optimization_options['epsilon'])
            # learning rate / step size
            self.learning_rate = float_type(optimization_options['learning_rate'])
            # weights for training files
            self._weights = optimization_options['weights']
            # maximum norm for parameter updates
            self._max_gradient_norm = float_type(
                optimization_options['max_gradient_norm'])
            # number of noise samples for sampling based output
            num_noise_samples = optimization_options['num_noise_samples']
            # noise sample sharing for sampling based output
            noise_sharing = optimization_options['noise_sharing']
        except KeyError as e:
            raise ValueError("Option {} is missing from optimization options."
                             .format(e))

        self._unk_id = self.network.vocabulary.word_to_id['<unk>']

        # The function takes as inputs a mini-batch of word IDs and class IDs,
        # and slices the input and target IDs for the network.
        batch_word_ids = tensor.matrix('optimizer/batch_word_ids',
                                       dtype='int64')
        batch_word_ids.tag.test_value = test_value(
            size=(101, 16), high=self.network.vocabulary.num_shortlist_words())
        batch_class_ids = tensor.matrix('optimizer/batch_class_ids',
                                        dtype='int64')
        batch_class_ids.tag.test_value = test_value(
            size=(101, 16), high=self.network.vocabulary.num_classes())

        # Derive the symbolic expression for updating the gradient with regard
        # to each parameter.
        cost, num_words = cost_function.get_tensor()
        self._gradients = \
            tensor.grad(cost, wrt=list(self.network.get_variables().values()))

        # The function takes as input the learning rate.
        alpha = tensor.scalar('optimizer/alpha',
                              dtype=theano.config.floatX)
        alpha.tag.test_value = 0.1

        # The function takes as input a matrix of weights, one for each
        # target word. These are used to scale the parameter updates.
        weights = tensor.matrix('optimizer/weights',
                                dtype=theano.config.floatX)
        weights.tag.test_value = test_value(size=(100, 16), high=1.0)
        word_positions = tensor.eq(self.network.mask, 1).nonzero()
        weight = weights[word_positions].sum()
        num_words_float = tensor.cast(num_words, theano.config.floatX)
        modified_alpha = tensor.switch(tensor.gt(num_words, 0),
                                       alpha * weight / num_words_float,
                                       alpha)

        # Ignore unused input, because is_training is only used by dropout
        # layer.
        self.update_function = theano.function(
            [batch_word_ids, batch_class_ids, self.network.mask, weights,
             alpha],
            [cost, num_words],
            givens=[(network.input_word_ids, batch_word_ids[:-1]),
                    (network.input_class_ids, batch_class_ids[:-1]),
                    (network.target_word_ids, batch_word_ids[1:]),
                    (network.target_class_ids, batch_class_ids[1:]),
                    (self.network.is_training, numpy.int8(1)),
                    (self.network.num_noise_samples,
                     numpy.int64(num_noise_samples))],
            updates=self._get_param_updates(alpha),
            name='update_function',
            on_unused_input='ignore',
            profile=profile)

    def get_state(self, state):
        """Pulls parameter values from Theano shared variables.

        If there already is a parameter in the state, it will be replaced, so it
        has to have the same number of elements.

        :type state: h5py.File
        :param state: HDF5 file for storing the optimization parameters
        """

        h5_optimizer = state.require_group('optimizer')
        h5_optimizer.attrs['learning_rate'] = self.learning_rate

        self._params.get_state(state)

    def set_state(self, state):
        """Sets the values of Theano shared variables.

        Requires that ``state`` contains values for all the optimization
        parameters.

        :type state: h5py.File
        :param state: HDF5 file that contains the optimization parameters
        """

        if 'optimizer' not in state:
            raise IncompatibleStateError("Optimizer state is missing.")
        h5_optimizer = state['optimizer']

        if 'learning_rate' not in h5_optimizer.attrs:
            raise IncompatibleStateError("Learning rate is missing from "
                                         "optimizer state.")
        self.learning_rate = h5_optimizer.attrs['learning_rate']

        self._params.set_state(state)

    def update_minibatch(self, word_ids, class_ids, file_ids, mask):
        """Optimizes the neural network parameters using the given inputs and
        learning rate.

        :type word_ids: ndarray of ints
        :param word_ids: a 2-dimensional matrix, indexed by time step and
                         sequence, that contains the word IDs

        :type class_ids: ndarray of ints
        :param class_ids: a 2-dimensional matrix, indexed by time step and
                          sequence, that contains the class IDs

        :type file_ids: ndarray of ints
        :param file_ids: a 2-dimensional matrix, indexed by time step and
                         sequence, that identifies the file in case of multiple
                         training files

        :type mask: numpy.ndarray of a floating point type
        :param mask: a 2-dimensional matrix, indexed by time step and sequence,
                     that masks out elements past the sequence ends.
        """

        # We should predict probabilities of the words at the following time
        # step.
        mask = mask[1:]
        file_ids = file_ids[1:]
        weights = self._weights[file_ids]
        alpha = self.learning_rate
        self.update_function(word_ids, class_ids, mask, weights, alpha)

    @abstractmethod
    def _get_param_updates(self, alpha):
        """Returns Theano expressions for updating the model parameters and any
        additional parameters required by the optimizer. Implemented by every
        optimizer subclass.

        :type alpha: Variable
        :param alpha: a scale to be applied to the model parameter updates

        :rtype: iterable over pairs (shared variable, new expression)
        :returns: expressions how to update the optimizer parameters
        """

        assert False

    def _normalize(self, deltas):
        """Normalizes the norm of parameter updates to given maximum value.

        :type deltas: dict of strs to symbolic tensors
        :param deltas: mapping from variable names to symbolic tensors that
                       describe the amount and direction of parameter updates
                       (normally the negative gradient of each parameter), after
                       any adaptation applied by the optimization method

        :rtype: dict of strs to symbolic tensors
        :returns: mapping from variable names to symbolic tensors that describe
                  ``deltas`` after normalization has been applied
        """

        max_norm = self._max_gradient_norm
        if max_norm is not None:
            norm = tensor.sqrt(sum_of_squares(deltas.values()))
            target_norm = tensor.clip(norm, 0.0, max_norm)
            for name, delta in deltas.items():
                deltas[name] = delta * target_norm / (self._epsilon + norm)
        return deltas
