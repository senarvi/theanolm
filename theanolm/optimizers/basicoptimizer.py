#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import abstractmethod, ABCMeta
from time import time
import logging
import numpy
import theano
import theano.tensor as tensor
from theanolm.exceptions import IncompatibleStateError, NumberError

class BasicOptimizer(object, metaclass=ABCMeta):
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

        float_type = numpy.dtype(theano.config.floatX).type

        # numerical stability / smoothing term to prevent divide-by-zero
        if not 'epsilon' in optimization_options:
            raise ValueError("'epsilon' is not given in optimization options.")
        self._epsilon = float_type(optimization_options['epsilon'])

        # learning rate / step size
        if not 'learning_rate' in optimization_options:
            raise ValueError("'learning_rate' is not given in optimization "
                             "options.")
        self.learning_rate = float_type(optimization_options['learning_rate'])

        # weights for training files
        if not 'weights' in optimization_options:
            raise ValueError("'weights' is not given in optimization options.")
        self._weights = optimization_options['weights']

        # maximum norm for parameter updates
        if not 'max_gradient_norm' in optimization_options:
            raise ValueError("'max_gradient_norm' is not given in optimization "
                             "options.")
        else:
            self._max_gradient_norm = float_type(
                optimization_options['max_gradient_norm'])

        # ignore <unk> tokens?
        if not 'ignore_unk' in optimization_options:
            raise ValueError("'ignore_unk' is not given in optimization "
                             "options.")
        ignore_unk = optimization_options['ignore_unk']

        # penalty for <unk> tokens
        if not 'unk_penalty' in optimization_options:
            raise ValueError("'unk_penalty' is not given in optimization "
                             "options.")
        unk_penalty = optimization_options['unk_penalty']

        unk_id = self.network.vocabulary.word_to_id['<unk>']

        # Derive the symbolic expression for log probability of each word.
        logprobs = tensor.log(self.network.prediction_probs)
        # If requested, predict <unk> with constant score.
        if not unk_penalty is None:
            unk_mask = tensor.eq(self.network.word_input[1:], unk_id)
            unk_indices = unk_mask.nonzero()
            logprobs = tensor.set_subtensor(logprobs[unk_indices], unk_penalty)
        # Ignore logprobs, when the next input word (the one predicted) is
        # masked out, and possibly also when  it is the <unk> token. The mask
        # has to be cast to floatX, otherwise the result will be float64 and
        # pulled out from the GPU earlier than necessary.
        mask = self.network.mask[1:]
        if ignore_unk:
            mask = mask * tensor.neq(self.network.word_input[1:], unk_id)
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
            updates=self._gradient_update_exprs(),
            name='gradient_update_function',
            on_unused_input='ignore',
            profile=profile)

        alpha = tensor.scalar('optimizer/update_weight',
                              dtype=theano.config.floatX)
        self.model_update_function = theano.function(
            [alpha],
            [],
            updates=self._model_update_exprs(alpha),
            name='model_update_function',
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

        if not 'optimizer' in state:
            raise IncompatibleStateError("Optimizer state is missing.")
        h5_optimizer = state['optimizer']

        if not 'learning_rate' in h5_optimizer.attrs:
            raise IncompatibleStateError("Learning rate is missing from "
                                         "optimizer state.")
        self.learning_rate = h5_optimizer.attrs['learning_rate']

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

        update_start_time = time()

        self.update_cost = self.gradient_update_function(word_ids, class_ids,
                                                         mask)
        if numpy.isnan(self.update_cost) or numpy.isinf(self.update_cost):
            raise NumberError("Mini-batch cost computation resulted in a "
                              "numerical error.")

        alpha = self.learning_rate
        num_words = numpy.count_nonzero(mask)
        float_type = numpy.dtype(theano.config.floatX).type
        if num_words > 0:
            weights = self._weights[file_ids]
            alpha *= weights[mask == 1].sum() / float_type(num_words)
        self.model_update_function(alpha)

        self.update_duration = time() - update_start_time

    @abstractmethod
    def _gradient_update_exprs(self):
        """Returns Theano expressions for updating the any gradient variables
        needed by the optimizer. Implemented by every optimizer subclass.

        :rtype: iterable over pairs (shared variable, new expression)
        :returns: expressions how to update the gradient variables
        """

        return

    @abstractmethod
    def _model_update_exprs(self, alpha):
        """Returns Theano expressions for updating the model parameter, given
        the gradient expressions. Implemented by every optimizer subclass.

        :type alpha: TensorVariable
        :param alpha: a scale to be applied to the parameter updates

        :rtype: iterable over pairs (shared variable, new expression)
        :returns: expressions how to update the model parameters
        """

        return

    def _normalize(self, updates):
        """Normalizes the norm of a parameter update to given maximum value.

        :type updates: dict of str to TensorVariable
        :param updates: dictionary of symbolic variables that describe the
                        negative gradient of each parameter, after any
                        optimization method specific adaptation

        :rtype: dict of str to TensorVariable
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
