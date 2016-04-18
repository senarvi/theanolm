#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import logging
import numpy
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theanolm.exceptions import IncompatibleStateError, InputError
from theanolm.layers import create_layer, NetworkInput
from theanolm.matrixfunctions import test_value

class Network(object):
    """Neural Network

    A class that stores the neural network architecture and state.
    """

    def __init__(self, vocabulary, architecture, batch_processing=True, profile=False):
        """Initializes the neural network parameters for all layers, and
        creates Theano shared variables from them.

        :type vocabulary: Vocabulary
        :param vocabulary: mapping between word IDs and word classes

        :type architecture: Architecture
        :param architecture: an object that describes the network architecture

        :type batch_processing: bool
        :param batch_processing: True creates a network for processing
                                 mini-batches, False creates a network for
                                 progressing one time step at a time

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        self.vocabulary = vocabulary
        self.architecture = architecture
        self.batch_processing = batch_processing

        M1 = 2147483647
        M2 = 2147462579
        random_seed = [
            numpy.random.randint(0, M1),
            numpy.random.randint(0, M1),
            numpy.random.randint(1, M1),
            numpy.random.randint(0, M2),
            numpy.random.randint(0, M2),
            numpy.random.randint(1, M2)]
        self.random = RandomStreams(random_seed)

        # Word and class inputs will be available to NetworkInput layers.
        self.word_input = tensor.matrix('network/word_input', dtype='int64')
        self.word_input.tag.test_value = test_value(
            size=(100, 16),
            max_value=vocabulary.num_words())
        self.class_input = tensor.matrix('network/class_input', dtype='int64')
        self.class_input.tag.test_value = test_value(
            size=(100, 16),
            max_value=vocabulary.num_classes())

        # Recurrent layers will create these lists, used by TextSampler to
        # initialize state variables of appropriate sizes.
        self.recurrent_state_input = []
        self.recurrent_state_size = []

        # Create the layers.
        logging.debug("Creating layers.")
        self.layers = OrderedDict()
        for input_options in architecture.inputs:
            input = NetworkInput(input_options, self)
            self.layers[input.name] = input
        for layer_description in architecture.layers:
            layer_options = dict()
            for variable, value in layer_description.items():
                if variable == 'inputs':
                    layer_options['input_layers'] = \
                        [self.layers[x] for x in value]
                else:
                    layer_options[variable] = value
            if layer_options['name'] == architecture.output_layer:
                layer_options['size'] = vocabulary.num_classes()
            layer = create_layer(layer_options, self, profile=profile)
            self.layers[layer.name] = layer
        self.output_layer = self.layers[architecture.output_layer]

        # This list will be filled by the recurrent layers to contain the
        # recurrent state outputs, required by TextSampler.
        self.recurrent_state_output = [None] * len(self.recurrent_state_size)

        # Create initial parameter values.
        logging.debug("Initializing parameters.")
        self.param_init_values = OrderedDict()
        num_params = 0
        for layer in self.layers.values():
            for name, value in layer.param_init_values.items():
                logging.debug("- %s size=%d", name, value.size)
                num_params += value.size
            self.param_init_values.update(layer.param_init_values)
        logging.debug("Total number of parameters: %d", num_params)

        # Create Theano shared variables.
        self.params = {name: theano.shared(value, name)
                       for name, value in self.param_init_values.items()}
        for layer in self.layers.values():
            layer.set_params(self.params)

        if batch_processing:
            self.create_batch_structure()
        else:
            self.create_onestep_structure()

    def create_batch_structure(self):
        """Creates the network structure for mini-batch processing.

        Creates the symbolic matrix self.output, which describes the output
        probability of the next input word at each time step and sequence. The
        shape will be the same as that of self.input, except that it will
        contain one less time step.
        """

        # mask is used to mask out the rest of the input matrix, when a sequence
        # is shorter than the maximum sequence length. The mask is kept as int8
        # data type, which is how Tensor stores booleans.
        self.mask = tensor.matrix('network/mask', dtype='int8')
        self.mask.tag.test_value = test_value(
            size=(100, 16),
            max_value=True)

        # Dropout layer needs to know whether we are training or evaluating.
        self.is_training = tensor.scalar('network/is_training', dtype='int8')
        self.is_training.tag.test_value = 1

        for layer in self.layers.values():
            layer.create_structure()

        self.output = self.output_layer.output

        # The input at the next time step is what the output (predicted word)
        # should be.
        class_ids = self.class_input[1:].flatten()
        output_probs = self.output[:-1].flatten()

        # An index to a flattened input matrix times the vocabulary size can be
        # used to index the same location in the output matrix. The class ID is
        # added to index the probability of that word.
        target_indices = \
            tensor.arange(class_ids.shape[0]) * self.vocabulary.num_classes() \
            + class_ids
        target_probs = output_probs[target_indices]

        # Reshape to a matrix. Now we have one less time step.
        num_time_steps = self.class_input.shape[0] - 1
        num_sequences = self.class_input.shape[1]
        self.prediction_probs = target_probs.reshape(
            [num_time_steps, num_sequences])

    def create_onestep_structure(self):
        """Creates the network structure for one-step processing.
        """

        # Create a mask for the case where we have only one word ID.
        self.mask = tensor.alloc(numpy.int8(1), 1, 1)

        # Dropout layer needs to know whether we are training or evaluating.
        self.is_training = tensor.scalar('network/is_training', dtype='int8')
        self.is_training.tag.test_value = 1

        for layer in self.layers.values():
            layer.create_structure()

        self.output = self.output_layer.output

    def get_state(self, state):
        """Pulls parameter values from Theano shared variables.

        If there already is a parameter in the state, it will be replaced, so it
        has to have the same number of elements.

        :type state: h5py.File
        :param state: HDF5 file for storing the neural network parameters
        """

        for name, param in self.params.items():
            if name in state:
                state[name][:] = param.get_value()
            else:
                state.create_dataset(name, data=param.get_value())

        self.architecture.get_state(state)

    def set_state(self, state):
        """Sets the values of Theano shared variables.

        Requires that ``state`` contains values for all the neural network
        parameters.

        :type state: h5py.File
        :param state: HDF5 file that contains the neural network parameters
        """

        for name, param in self.params.items():
            if not name in state:
                raise IncompatibleStateError(
                    "Parameter %s is missing from neural network state." % name)
            new_value = state[name].value
            param.set_value(new_value)
            if len(new_value.shape) == 0:
                logging.debug("%s <- %s", name, str(new_value))
            else:
                logging.debug("%s <- array%s", name, str(new_value.shape))
        try:
            self.architecture.check_state(state)
        except IncompatibleStateError as error:
            raise IncompatibleStateError(
                "Attempting to restore state of a network that is incompatible "
                "with this architecture. " + str(error))

    def add_recurrent_state(self, size):
        """Adds a recurrent state variable and returns its index.

        Used by recurrent layers to add a state variable that has to be passed
        from one time step to the next, when generating text using one-step
        processing.
        """

        index = len(self.recurrent_state_size)
        assert index == len(self.recurrent_state_input)

        # The variables are in the structure of a mini-batch (3-dimensional
        # array) to keep the layer functions general.
        variable = tensor.tensor3('network/recurrent_state_' + str(index),
                                  dtype=theano.config.floatX)
        variable.tag.test_value = test_value(size=(1, 1, size), max_value=1.0)

        self.recurrent_state_size.append(size)
        self.recurrent_state_input.append(variable)

        return index
