#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import logging
import numpy
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theanolm.exceptions import IncompatibleStateError, InputError
from theanolm.network.networkinput import NetworkInput
from theanolm.network.projectionlayer import ProjectionLayer
from theanolm.network.tanhlayer import TanhLayer
from theanolm.network.grulayer import GRULayer
from theanolm.network.lstmlayer import LSTMLayer
from theanolm.network.softmaxlayer import SoftmaxLayer
from theanolm.network.hsoftmaxlayer import HSoftmaxLayer
from theanolm.network.dropoutlayer import DropoutLayer
from theanolm.matrixfunctions import test_value

def create_layer(layer_options, *args, **kwargs):
    """Constructs one of the Layer classes based on a layer definition.

    :type layer_type: str
    :param layer_type: a text string describing the layer type
    """

    layer_type = layer_options['type']
    if layer_type == 'projection':
        return ProjectionLayer(layer_options, *args, **kwargs)
    elif layer_type == 'tanh':
        return TanhLayer(layer_options, *args, **kwargs)
    elif layer_type == 'lstm':
        return LSTMLayer(layer_options, *args, **kwargs)
    elif layer_type == 'gru':
        return GRULayer(layer_options, *args, **kwargs)
    elif layer_type == 'softmax':
        return SoftmaxLayer(layer_options, *args, **kwargs)
    elif layer_type == 'hsoftmax':
        return HSoftmaxLayer(layer_options, *args, **kwargs)
    elif layer_type == 'dropout':
        return DropoutLayer(layer_options, *args, **kwargs)
    else:
        raise ValueError("Invalid layer type requested: " + layer_type)

class Network(object):
    """Neural Network

    A class that stores the neural network architecture and state.
    """

    def __init__(self, vocabulary, architecture,
                 predict_next_distribution=False, profile=False):
        """Initializes the neural network parameters for all layers, and
        creates Theano shared variables from them.

        :type vocabulary: Vocabulary
        :param vocabulary: mapping between word IDs and word classes

        :type architecture: Architecture
        :param architecture: an object that describes the network architecture

        :type predict_next_distribution: bool
        :param predict_next_distribution: if set to True, creates a network that
            produces the probability distribution for the next word (instead of
            of target probabilities for a mini-batch)

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        self.vocabulary = vocabulary
        self.architecture = architecture
        self.predict_next_distribution = predict_next_distribution

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

        # Recurrent layers will create these lists, used to initialize state
        # variables of appropriate sizes, for doing forward passes one step at a
        # time.
        self.recurrent_state_input = []
        self.recurrent_state_size = []

        # Create the layers.
        logging.debug("Creating layers.")
        self.layers = OrderedDict()
        for input_options in architecture.inputs:
            input = NetworkInput(input_options, self)
            self.layers[input.name] = input
        for layer_description in architecture.layers:
            layer_options = self._layer_options_from_description(
                layer_description)
            if layer_options['name'] == architecture.output_layer:
                layer_options['size'] = vocabulary.num_classes()
            layer = create_layer(layer_options, self, profile=profile)
            self.layers[layer.name] = layer
        self.output_layer = self.layers[architecture.output_layer]

        # This list will be filled by the recurrent layers to contain the
        # recurrent state outputs, for doing forward passes one step at a time.
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

        # mask is used to mask out the rest of the input matrix, when a sequence
        # is shorter than the maximum sequence length. The mask is kept as int8
        # data type, which is how Tensor stores booleans. When the network is
        # used to predict the probability distribution of the next word, the
        # matrix contains only one word ID.
        if self.predict_next_distribution:
            self.mask = tensor.alloc(numpy.int8(1), 1, 1)
        else:
            self.mask = tensor.matrix('network/mask', dtype='int8')
            self.mask.tag.test_value = test_value(
                size=(100, 16),
                max_value=True)

        # Dropout layer needs to know whether we are training or evaluating.
        self.is_training = tensor.scalar('network/is_training', dtype='int8')
        self.is_training.tag.test_value = 1

        for layer in self.layers.values():
            layer.create_structure()

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
        from one time step to the next, when generating text or computing
        lattice probabilities.
        """

        index = len(self.recurrent_state_size)
        assert index == len(self.recurrent_state_input)

        # The variables are in the structure of a mini-batch (3-dimensional
        # array) to keep the layer functions general.
        variable = tensor.tensor3('network/recurrent_state_' + str(index),
                                  dtype=theano.config.floatX)
        variable.tag.test_value = test_value(size=(1, 3, size), max_value=1.0)

        self.recurrent_state_size.append(size)
        self.recurrent_state_input.append(variable)

        return index

    def output_probs(self):
        """Returns the output probabilities for the whole vocabulary.

        The network has to have ``predict_next_distribution == True``.

        :type mask: TensorVariable
        :param mask: a symbolic 3-dimensional matrix that contains a probability
                     for each time step (except the last), each sequence, and
                     each word.
        """

        if not hasattr(self.output_layer, 'output_probs'):
            raise RuntimeError("The final layer is not an output layer.")
        if self.output_layer.output_probs is None:
            raise RuntimeError("Trying to read all output probabilities, while "
                               "the output layer is configured to produce only "
                               "target probabilities.")
        return self.output_layer.output_probs

    def target_probs(self):
        """Returns the output probabilities for the predicted words, i.e. the
        following word at each time step.

        The network has to have ``predict_next_distribution == False``.

        :type mask: TensorVariable
        :param mask: a symbolic 2-dimensional matrix that contains a probability
                     for each time step (except the last) and each sequence
        """

        if not hasattr(self.output_layer, 'target_probs'):
            raise RuntimeError("The final layer is not an output layer.")
        if self.output_layer.target_probs is None:
            raise RuntimeError("Trying to read target probabilities, while the "
                               "output layer is configured to produce all "
                               "probabilities.")
        return self.output_layer.target_probs

    def _layer_options_from_description(self, description):
        """Creates layer options based on textual architecture description.

        Most of the fields in a layer description are kept as strings. The field
        ``input_layers`` is converted to a list of actual layers found from
        ``self.layers``.

        :type description: dict
        :param description: dictionary of textual layer fields

        :rtype: dict
        :result: layer options
        """

        result = dict()
        for variable, value in description.items():
            if variable == 'inputs':
                try:
                    result['input_layers'] = [self.layers[x] for x in value]
                except KeyError as e:
                    raise InputError("Input layer `{}' does not exist, when "
                                     "creating layer `{}'.".format(
                                     e.args[0],
                                     description['name']))
            else:
                result[variable] = value
        return result
