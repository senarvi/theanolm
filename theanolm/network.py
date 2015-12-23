#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import logging
import numpy
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theanolm.exceptions import IncompatibleStateError, InputError
from theanolm.layers import *
from theanolm.matrixfunctions import test_value

class Network(object):
    """Neural Network

    A class that stores the neural network architecture and state.
    """

    class Architecture(object):
        """Neural Network Architecture Description
        """

        def __init__(self, layers):
            """Constructs a description of the neural network architecture.

            :type layers: list of dict
            :param layers: options for each layer as a list of dictionaries
            """

            self.layers = layers

        @classmethod
        def from_state(classname, state):
            """Constructs a description of the network architecture stored in a
            state.

            :type state: dict of numpy types
            :param state: a dictionary of the architecture parameters
            """

            if not 'arch.layers' in state:
                raise IncompatibleStateError(
                    "Parameter 'arch.layers' is missing from neural network state.")
            # An ugly workaround to be able to save arbitrary data in a .npz file.
            try:
                dummy_dict = state['arch.layers'][()]
            except KeyError:
                dummy_dict = state['arch.layers']
            layers = dummy_dict['data']
            return classname(layers)

        @classmethod
        def from_description(classname, description_file):
            """Reads a description of the network architecture from a text file.

            :type description_file: file or file-like object
            :param description_file: text file containing the description

            :rtype: Network.Architecture
            :returns: an object describing the network architecture
            """

            layers = []

            for line in description_file:
                layer_description = dict()

                fields = line.split()
                if not fields:
                    continue
                if fields[0] != 'layer':
                    raise InputError("Unknown network element: {}.".format(
                        fields[0]))
                for field in fields[1:]:
                    variable, value = field.split('=')
                    if variable == 'type':
                        layer_description['type'] = value
                    elif variable == 'name':
                        layer_description['name'] = value
                    elif variable == 'input':
                        if 'inputs' in layer_description:
                            layer_description['inputs'].append(value)
                        else:
                            layer_description['inputs'] = [value]
                    elif variable == 'output':
                        layer_description['output'] = value

                if not 'type' in layer_description:
                    raise InputError("'type' is not given in a layer description.")
                if not 'name' in layer_description:
                    raise InputError("'name' is not given in a layer description.")
                if not 'inputs' in layer_description:
                    raise InputError("'input' is not given in a layer description.")
                # Output size is not required. By default it is the same as input size.
                if not 'output' in layer_description:
                    layer_description['output'] = '-'

                layers.append(layer_description)

            return classname(layers)

        def get_state(self):
            """Returns a dictionary of parameters that should be saved along
            with the network state.

            For consistency, all the parameter values are returned as numpy
            types, since state read from a model file also contains numpy types.

            :rtype: dict of numpy types
            :returns: a dictionary of the architecture parameters
            """

            result = OrderedDict()
            result['arch.layers'] = { 'data': self.layers }
            return result

        def check_state(self, state):
            """Checks that the architecture stored in a state matches this
            network architecture, and raises an ``IncompatibleStateError``
            if not.

            :type state: dict of numpy types
            :param state: dictionary of neural network parameters
            """

            if not 'arch.layers' in state:
                raise IncompatibleStateError(
                    "Parameter 'arch.layers' is missing from neural network state.")
            # An ugly workaround to be able to save arbitrary data in a .npz file.
            try:
                dummy_dict = state['arch.layers'][()]
            except KeyError:
                dummy_dict = state['arch.layers']
            state_layers = dummy_dict['data']
            for layer1, layer2 in zip(self.layers, state_layers):
                if layer1['type'] != layer2['type']:
                    raise IncompatibleStateError(
                        "Neural network state has {0}={1}, while this architecture "
                        "has {0}={2}.".format('type', layer2['type'], layer1['type']))
                if layer1['name'] != layer2['name']:
                    raise IncompatibleStateError(
                        "Neural network state has {0}={1}, while this architecture "
                        "has {0}={2}.".format('name', layer2['name'], layer1['name']))
                if layer1['output'] != layer2['output']:
                    raise IncompatibleStateError(
                        "Neural network state has {0}={1}, while this architecture "
                        "has {0}={2}.".format('output', layer2['output'], layer1['output']))

    def __init__(self, dictionary, architecture, batch_processing=True, profile=False):
        """Initializes the neural network parameters for all layers, and
        creates Theano shared variables from them.

        :type dictionary: Dictionary
        :param dictionary: mapping between word IDs and word classes

        :type architecture: Network.Architecture
        :param architecture: an object that describes the network architecture

        :type batch_processing: bool
        :param batch_processing: True creates a network for processing
                                 mini-batches, False creates a network for
                                 progressing one time step at a time

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        self.dictionary = dictionary
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

        # Recurrent layers will create these lists, used by TextSampler to
        # initialize state variables of appropriate sizes.
        self.recurrent_state_input = []
        self.recurrent_state_size = []

        # Create the layers.
        logging.debug("Creating layers.")
        self.network_input = NetworkInput(dictionary.num_classes(), self)
        self.output_layer = None
        self.layers = OrderedDict()
        self.layers['X'] = self.network_input
        for layer_description in architecture.layers:
            layer_type = layer_description['type']
            layer_name = layer_description['name']
            inputs = [ self.layers[x] for x in layer_description['inputs'] ]
            if layer_description['output'] == 'Y':
                output_size = dictionary.num_classes()
            elif layer_description['output'] == '-':
                output_size = sum([ x.output_size for x in inputs ])
            else:
                output_size = int(layer_description['output'])
            self.layers[layer_name] = create_layer(layer_type,
                                                   layer_name,
                                                   inputs,
                                                   output_size,
                                                   self,
                                                   profile=profile)
            if layer_description['output'] == 'Y':
                self.output_layer = self.layers[layer_name]
        if self.output_layer is None:
            raise InputError("None of the layers in architecture description "
                             "have 'output=Y'.")

        # This list will be filled by the recurrent layers to contain the
        # recurrent state outputs, required by TextSampler.
        self.recurrent_state_output = [None] * len(self.recurrent_state_size)

        # Create initial parameter values.
        self.param_init_values = OrderedDict()
        for layer in self.layers.values():
            self.param_init_values.update(layer.param_init_values)

        # Create Theano shared variables.
        self.params = { name: theano.shared(value, name)
                        for name, value in self.param_init_values.items() }
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
        self.mask = tensor.matrix('network.mask', dtype='int8')
        self.mask.tag.test_value = test_value(
            size=(100, 16),
            max_value=True)

        # Dropout layer needs to know whether we are training or evaluating.
        self.is_training = tensor.scalar('network.is_training', dtype='int8')
        self.is_training.tag.test_value = 1

        for layer in self.layers.values():
            layer.create_structure()

        self.input = self.network_input.output
        self.output = self.output_layer.output

        # The input at the next time step is what the output (predicted word)
        # should be.
        word_ids = self.input[1:].flatten()
        output_probs = self.output[:-1].flatten()

        # An index to a flattened input matrix times the vocabulary size can be
        # used to index the same location in the output matrix. The word ID is
        # added to index the probability of that word.
        target_indices = \
            tensor.arange(word_ids.shape[0]) * self.dictionary.num_classes() \
            + word_ids
        target_probs = output_probs[target_indices]

        # Reshape to a matrix. Now we have one less time step.
        num_time_steps = self.input.shape[0] - 1
        num_sequences = self.input.shape[1]
        self.prediction_probs = target_probs.reshape(
            [num_time_steps, num_sequences])

    def create_onestep_structure(self):
        """Creates the network structure for one-step processing.
        """

        # Create a mask for the case where we have only one word ID.
        self.mask = tensor.alloc(numpy.int8(1), 1, 1)

        # Dropout layer needs to know whether we are training or evaluating.
        self.is_training = tensor.scalar('network.is_training', dtype='int8')
        self.is_training.tag.test_value = 1

        for layer in self.layers.values():
            layer.create_structure()

        self.input = self.network_input.output
        self.output = self.output_layer.output

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
            new_value = state[name]
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
        variable = tensor.tensor3('network.recurrent_state_' + str(index),
                                  dtype=theano.config.floatX)
        variable.tag.test_value = test_value(size=(1, 1, size), max_value=1.0)

        self.recurrent_state_size.append(size)
        self.recurrent_state_input.append(variable)

        return index
