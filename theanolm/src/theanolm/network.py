#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.exceptions import IncompatibleStateError
from theanolm.layers import *
from theanolm.matrixfunctions import test_value

class Network(object):
    """Neural Network

    A class that stores the neural network architecture and state.
    """

    class Architecture(object):
        """Neural Network Architecture
        """

        def __init__(self, word_projection_dim, hidden_layer_type,
                     hidden_layer_size, skip_layer_size):
            """Constructs a description of the specified network architecture.

            :type word_projection_dim: int
            :param word_projection_dim: dimensionality of the word projections

            :type hidden_layer_type: str
            :param hidden_layer_type: type of the units used in the hidden layer

            :type hidden_layer_size: int
            :param hidden_layer_size: number of outputs in the hidden layer

            :type skip_layer_size: int
            :param skip_layer_size: number of outputs in the skip-layer, or 0
                                    for no skip-layer
            """

            self.word_projection_dim = word_projection_dim
            self.hidden_layer_type = hidden_layer_type
            self.hidden_layer_size = hidden_layer_size
            self.skip_layer_size = skip_layer_size

        @classmethod
        def from_state(classname, state):
            """Constructs a description of the network architecture stored in a
            state.

            :type state: dict of numpy types
            :param state: a dictionary of the architecture parameters
            """

            classname._check_parameter_in_state('arch.word_projection_dim', state)
            classname._check_parameter_in_state('arch.hidden_layer_type', state)
            classname._check_parameter_in_state('arch.hidden_layer_size', state)
            classname._check_parameter_in_state('arch.skip_layer_size', state)
            return classname(
                state['arch.word_projection_dim'].item(),
                state['arch.hidden_layer_type'].item(),
                state['arch.hidden_layer_size'].item(),
                state['arch.skip_layer_size'].item())

        def __str__(self):
            """Returns a string representation of the architecture for printing
            to the user.

            :rtype: str
            :returns: a string describing the architecture
            """

            result = "Word projection dimensionality: "
            result += str(self.word_projection_dim)
            result += "\nHidden layer type: "
            result += str(self.hidden_layer_type)
            result += "\nHidden layer size: "
            result += str(self.hidden_layer_size)
            result += "\nSkip layer size: "
            result += str(self.skip_layer_size)
            return(result)

        def get_state(self):
            """Returns a dictionary of parameters that should be saved along
            with the network state.

            For consistency, all the parameter values are returned as numpy
            types, since state read from a model file also contains numpy types.

            :rtype: dict of numpy types
            :returns: a dictionary of the architecture parameters
            """

            result = OrderedDict()
            result['arch.word_projection_dim'] = numpy.int64(self.word_projection_dim)
            result['arch.hidden_layer_type'] = numpy.str_(self.hidden_layer_type)
            result['arch.hidden_layer_size'] = numpy.int64(self.hidden_layer_size)
            result['arch.skip_layer_size'] = numpy.int64(self.skip_layer_size)
            return result

        def check_state(self, state):
            """Checks that the architecture stored in a state matches this
            network architecture, and raises an ``IncompatibleStateError``
            if not.

            :type state: dict of numpy types
            :param state: dictionary of neural network parameters
            """

            self._check_parameter_value(
                'arch.word_projection_dim', state, self.word_projection_dim)
            self._check_parameter_value(
                'arch.hidden_layer_type', state, self.hidden_layer_type)
            self._check_parameter_value(
                'arch.hidden_layer_size', state, self.hidden_layer_size)
            self._check_parameter_value(
                'arch.skip_layer_size', state, self.skip_layer_size)

        def _check_parameter_value(self, name, state, current_value):
            """Checks that the parameter value stored in a state matches the
            current value, and raises an ``IncompatibleStateError`` if not.

            :type name: str
            :param name: the parameter key in the state

            :type state: dict
            :param state: dictionary of neural network parameters
            """

            Network.Architecture._check_parameter_in_state(name, state)
            if state[name] != current_value:
                raise IncompatibleStateError(
                    "Neural network state has {0}={1}, while this architecture "
                    "has {0}={2}.".format(name, state[name], current_value))

        @staticmethod
        def _check_parameter_in_state(name, state):
            """Checks that the parameter value is stored in a state, and raises
            an ``IncompatibleStateError`` if not.

            :type name: str
            :param name: the parameter key in the state

            :type state: dict
            :param state: dictionary of neural network parameters
            """

            if not name in state:
                raise IncompatibleStateError(
                    "Parameter {0} is missing from neural network state."
                    "".format(name))


    def __init__(self, dictionary, architecture, profile=False):
        """Initializes the neural network parameters for all layers, and
        creates Theano shared variables from them.

        :type dictionary: Dictionary
        :param dictionary: mapping between word IDs and word classes

        :type architecture: Network.Architecture
        :param architecture: an object that describes the network architecture

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        self.dictionary = dictionary
        self.architecture = architecture

        # Create the layers.
        self.projection_layer = ProjectionLayer(
            dictionary.num_classes(),
            self.architecture.word_projection_dim)
        if self.architecture.hidden_layer_type == 'lstm':
            self.hidden_layer = LSTMLayer(
                self.architecture.word_projection_dim,
                self.architecture.hidden_layer_size,
                profile)
        elif self.architecture.hidden_layer_type == 'ss-lstm':
            self.hidden_layer = SSLSTMLayer(
                self.architecture.word_projection_dim,
                self.architecture.hidden_layer_size,
                profile)
        elif self.architecture.hidden_layer_type == 'gru':
            self.hidden_layer = GRULayer(
                self.architecture.word_projection_dim,
                self.architecture.hidden_layer_size,
                profile)
        else:
            raise ValueError("Invalid hidden layer type: " + \
                             self.architecture.hidden_layer_type)
        if self.architecture.skip_layer_size > 0:
            self.skip_layer = SkipLayer(
                self.architecture.hidden_layer_size,
                self.architecture.word_projection_dim,
                self.architecture.skip_layer_size)
            self.output_layer = OutputLayer(
                self.architecture.skip_layer_size,
                dictionary.num_classes())
        else:
            self.output_layer = OutputLayer(
                self.architecture.hidden_layer_size,
                dictionary.num_classes())

        # Create initial parameter values.
        self.param_init_values = OrderedDict()
        self.param_init_values.update(self.projection_layer.param_init_values)
        self.param_init_values.update(self.hidden_layer.param_init_values)
        if self.architecture.skip_layer_size > 0:
            self.param_init_values.update(self.skip_layer.param_init_values)
        self.param_init_values.update(self.output_layer.param_init_values)

        # Create Theano shared variables.
        self.params = {name: theano.shared(value, name)
                       for name, value in self.param_init_values.items()}

        self._create_minibatch_structure()
        self._create_onestep_structure()

    def _create_minibatch_structure(self):
        """Creates the network structure for mini-batch processing.

        Sets self.minibatch_output to a symbolic matrix, the same shape as
        self.minibatch_input, containing the output word probabilities, and
        self.minibatch_costs containing the negative log probabilities.
        """

        # minibatch_input describes the input matrix containing
        # [ number of time steps * number of sequences ] word IDs.
        self.minibatch_input = tensor.matrix('minibatch_input', dtype='int64')
        self.minibatch_input.tag.test_value = test_value(
            size=(100, 16),
            max_value=self.dictionary.num_classes())

        # mask is used to mask out the rest of the input matrix, when a sequence
        # is shorter than the maximum sequence length.
        self.minibatch_mask = \
            tensor.matrix('minibatch_mask', dtype=theano.config.floatX)
        self.minibatch_mask.tag.test_value = test_value(
            size=(100, 16),
            max_value=1.0)

        self.projection_layer.create_minibatch_structure(
            self.params,
            self.minibatch_input)
        self.hidden_layer.create_minibatch_structure(
            self.params,
            self.projection_layer.minibatch_output,
            mask=self.minibatch_mask)
        if self.architecture.skip_layer_size > 0:
            self.skip_layer.create_structure(
                self.params,
                self.hidden_layer.minibatch_output,
                self.projection_layer.minibatch_output)
            self.output_layer.create_minibatch_structure(
                self.params,
                self.skip_layer.output)
        else:
            self.output_layer.create_minibatch_structure(
                self.params,
                self.hidden_layer.minibatch_output)

        self.minibatch_output = self.output_layer.minibatch_output

        # Projection layer delays the output by one time step, so that the input
        # at each time step is what the output (predicted word) should be. Input
        # word IDs + the index times vocabulary size can be used to index a
        # flattened output matrix to read the probabilities of the correct
        # outputs.
        input_flat = self.minibatch_input.flatten()
        correct_output_indices = \
            tensor.arange(input_flat.shape[0]) * self.dictionary.num_classes() \
            + input_flat
        correct_output_probs = \
            self.minibatch_output.flatten()[correct_output_indices]
        correct_output_probs = correct_output_probs.reshape(
            [self.minibatch_input.shape[0], self.minibatch_input.shape[1]])
        self.prediction_probs = correct_output_probs

    def _create_onestep_structure(self):
        """Creates the network structure for one-step processing.
        """

        # onestep_input describes the input vector containing as many word IDs
        # as there are sequences (only one for the text sampler).
        self.onestep_input = tensor.vector('onestep_input', dtype='int64')
        self.onestep_input.tag.test_value = test_value(
            size=16,
            max_value=self.dictionary.num_classes())

        # onestep_state describes the state outputs of the previous time step
        # of the hidden layer. GRU has one state output, LSTM has two.
        self.onestep_state = \
            [tensor.matrix('onestep_state_' + str(i),
                           dtype=theano.config.floatX)
             for i in range(self.hidden_layer.num_state_variables)]
        for state_variable in self.onestep_state:
            state_variable.tag.test_value = test_value(
                size=(1, self.architecture.hidden_layer_size),
                max_value=1.0)

        self.projection_layer.create_onestep_structure(
            self.params,
            self.onestep_input)
        self.hidden_layer.create_onestep_structure(
            self.params,
            self.projection_layer.onestep_output,
            self.onestep_state)
        # The last state output from the hidden layer is the hidden state to be
        # passed on the the next layer.
        hidden_state_output = self.hidden_layer.onestep_outputs[-1]
        if self.architecture.skip_layer_size > 0:
            self.skip_layer.create_structure(
                self.params,
                hidden_state_output,
                self.projection_layer.onestep_output)
            self.output_layer.create_onestep_structure(
                self.params,
                self.skip_layer.output)
        else:
            self.output_layer.create_onestep_structure(
                self.params,
                hidden_state_output)

        self.onestep_output = self.output_layer.onestep_output

    def get_state(self):
        """Pulls parameter values from Theano shared variables.

        For consistency, all the parameter values are returned as numpy
        types, since state read from a model file also contains numpy types.

        :rtype: dict of numpy types
        :returns: a dictionary of the parameter values and network architecture
        """

        result = OrderedDict()
        for name, param in self.params.items():
            result[name] = param.get_value()
        result.update(self.architecture.get_state())
        return result

    def set_state(self, state):
        """Sets the values of Theano shared variables.

        Requires that ``state`` contains values for all the neural network
        parameters.

        :type state: dict of numpy types
        :param state: a dictionary of neural network parameters
        """

        for name, param in self.params.items():
            if not name in state:
                raise IncompatibleStateError(
                    "Parameter %s is missing from neural network state." % name)
            param.set_value(state[name])
        try:
            self.architecture.check_state(state)
        except IncompatibleStateError as error:
            raise IncompatibleStateError(
                "Attempting to restore state of a network that is incompatible "
                "with this architecture. " + str(error))

    def sequences_to_minibatch(self, sequences):
        """Transposes a list of sequences and Prepares a mini-batch for input to the neural network by transposing
        a matrix of word ID sequences and creating a mask matrix.

        The first dimensions of the returned matrix word_ids will be the time
        step, i.e. the index to a word in a sequence. In other words, the first
        row will contain the first word ID of each sequence, the second row the
        second word ID of each sequence, and so on. The rest of the matrix will
        be filled with zeros.

        The other returned matrix, mask, is the same size as word_ids, and will
        contain zeros where word_ids contains word IDs, and ones elsewhere
        (after sequence end).

        :type sequences: list of lists
        :param sequences: list of sequences, each of which is a list of word
                          IDs

        :rtype: tuple of numpy matrices
        :returns: two matrices - one contains the word IDs of each sequence
                  (0 after the last word), and the other contains a mask that
                  ís 1 after the last word
        """

        num_sequences = len(sequences)
        sequence_lengths = [len(s) for s in sequences]
        batch_length = numpy.max(sequence_lengths) + 1

        word_ids = numpy.zeros((batch_length, num_sequences)).astype('int64')
        probs = numpy.zeros((batch_length, num_sequences))
        probs = probs.astype(theano.config.floatX)
        mask = numpy.zeros((batch_length, num_sequences))
        mask = mask.astype(theano.config.floatX)
        for i, sequence in enumerate(sequences):
            word_ids[:sequence_lengths[i], i] = sequence
            mask[:sequence_lengths[i] + 1, i] = 1.0

        return word_ids, mask
