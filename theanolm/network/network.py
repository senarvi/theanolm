#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the Network class.
"""

from collections import OrderedDict
import sys
import logging

import h5py
import numpy
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theanolm import Vocabulary
from theanolm.backend import IncompatibleStateError, InputError
from theanolm.backend import UniformDistribution, LogUniformDistribution
from theanolm.backend import MultinomialDistribution
from theanolm.backend import test_value
from theanolm.network.architecture import Architecture
from theanolm.network.networkinput import NetworkInput
from theanolm.network.projectionlayer import ProjectionLayer
from theanolm.network.fclayer import FullyConnectedLayer
from theanolm.network.additionlayer import AdditionLayer
from theanolm.network.grulayer import GRULayer
from theanolm.network.lstmlayer import LSTMLayer
from theanolm.network.highwaylayer import HighwayLayer
from theanolm.network.glulayer import GLULayer
from theanolm.network.softmaxlayer import SoftmaxLayer
from theanolm.network.hsoftmaxlayer import HSoftmaxLayer
from theanolm.network.dropoutlayer import DropoutLayer
from theanolm.network.bidirectionallayer import BidirectionalLayer

def create_layer(layer_options, *args, **kwargs):
    """Constructs one of the Layer classes based on a layer definition.

    :type layer_type: str
    :param layer_type: a text string describing the layer type
    """

    layer_type = layer_options['type']
    if layer_type == 'projection':
        return ProjectionLayer(layer_options, *args, **kwargs)
    elif layer_type == 'fc' or layer_type == 'tanh':
        return FullyConnectedLayer(layer_options, *args, **kwargs)
    elif layer_type == 'add':
        return AdditionLayer(layer_options, *args, **kwargs)
    elif layer_type == 'lstm':
        return LSTMLayer(layer_options, *args, **kwargs)
    elif layer_type == 'gru':
        return GRULayer(layer_options, *args, **kwargs)
    elif layer_type == 'blstm' or layer_type == 'bgru':
        return BidirectionalLayer(layer_options, *args, **kwargs)
    elif layer_type == 'highway' or layer_type == 'highwaytanh':
        return HighwayLayer(layer_options, *args, **kwargs)
    elif layer_type == 'glu':
        return GLULayer(layer_options, *args, **kwargs)
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

    A class that creates the actual neural network graph using Theano. Functions
    that train and apply the neural network can be created by passing the input
    and output variables to ``theano.function()``.
    """

    class Mode():
        """Network Mode Selection

        Enumeration of options for selecting network mode. This will create a
        slightly different output for different purposes.

          - ``minibatch``: Process mini-batches with multiple sequences and time
                           steps. The output is a matrix with one less time
                           steps containing the probabilities of the words at
                           the next time step.
        """
        def __init__(self, minibatch=True, nce=False):
            self.minibatch = minibatch
            self.nce = nce

    def __init__(self, architecture, vocabulary, class_prior_probs=None,
                 mode=None, exclude_unk=False, default_device=None,
                 profile=False):
        """Initializes the neural network parameters for all layers, and
        creates Theano shared variables from them.

        :type architecture: Architecture
        :param architecture: an object that describes the network architecture

        :type vocabulary: Vocabulary
        :param vocabulary: mapping between word IDs and word classes

        :type class_prior_probs: numpy.ndarray
        :param class_prior_probs: empirical (unigram) distribution of the output
                                  classes (only required for training)

        :type mode: Network.Mode
        :param mode: selects mini-batch or single time step processing

        :type exclude_unk: bool
        :param exclude_unk: if set to ``True``, set ``<unk>`` probability to
                            zero before normalizing the network outputs
                            (required to get exact normalization during
                            inference)

        :type default_device: str
        :param default_device: default device where to store the shared variables

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        self.vocabulary = vocabulary
        self.architecture = architecture
        self.mode = self.Mode() if mode is None else mode

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
        self.input_word_ids = tensor.matrix('network/input_word_ids',
                                            dtype='int64')
        self.input_class_ids = tensor.matrix('network/input_class_ids',
                                             dtype='int64')
        if self.mode.minibatch:
            self.input_word_ids.tag.test_value = test_value(
                size=(20, 4), high=vocabulary.num_shortlist_words())
            self.input_class_ids.tag.test_value = test_value(
                size=(20, 4), high=vocabulary.num_classes())
        else:
            self.input_word_ids.tag.test_value = test_value(
                size=(1, 4), high=vocabulary.num_shortlist_words())
            self.input_class_ids.tag.test_value = test_value(
                size=(1, 4), high=vocabulary.num_classes())

        # Should the output layer set the <unk> probability to zero? In that
        # case OOVs are not counted when computing perplexity.
        self.exclude_unk = exclude_unk

        # Default device for shared variables.
        self._default_device = default_device

        # During training, the output layer bias vector is initialized to the
        # unigram probabilities.
        self.class_prior_probs = class_prior_probs

        # A shortlist model adds these logprobs to OOS logprobs predicted by the
        # network.
        if vocabulary.has_unigram_probs():
            oos_logprobs = numpy.log(vocabulary.get_oos_probs())
            oos_logprobs = oos_logprobs.astype(theano.config.floatX)
            self.oos_logprobs = theano.shared(oos_logprobs,
                                              'network/oos_logprobs')
        else:
            self.oos_logprobs = None

        # Recurrent layers will create these lists, used to initialize state
        # variables of appropriate sizes, for doing forward passes one step at a
        # time.
        self.recurrent_state_input = []
        self.recurrent_state_size = []

        # Create the layers.
        logging.debug("Creating layers.")
        self.layers = OrderedDict()
        for input_options in architecture.inputs:
            network_input = NetworkInput(input_options, self)
            self.layers[network_input.name] = network_input
        for layer_description in architecture.layers:
            layer_options = self._layer_options_from_description(
                layer_description)
            if layer_options['name'] == architecture.output_layer:
                layer_options['size'] = vocabulary.num_classes()
            # 'devices' not in layer_options is for backward compatibility.
            # Remove at some point.
            if ('devices' not in layer_options) or (not layer_options['devices']):
                layer_options['devices'] = [default_device]
            layer = create_layer(layer_options, self, profile=profile)
            self.layers[layer.name] = layer
        self.output_layer = self.layers[architecture.output_layer]
        num_params = sum(layer.num_params() for layer in self.layers.values())
        logging.debug("Total number of parameters: %d", num_params)

        # This list will be filled by the recurrent layers to contain the
        # recurrent state outputs, for doing forward passes one step at a time.
        self.recurrent_state_output = [None] * len(self.recurrent_state_size)

        # This input variable can be used to specify the classes whose
        # probabilities will be computed, instead of the whole distribution.
        self.target_class_ids = tensor.matrix('network/target_class_ids',
                                              dtype='int64')
        if self.mode.minibatch:
            self.target_class_ids.tag.test_value = test_value(
                size=(20, 4), high=vocabulary.num_classes())
        else:
            self.target_class_ids.tag.test_value = test_value(
                size=(1, 4), high=vocabulary.num_classes())

        # This input variable is used only for detecting <unk> target words.
        self.target_word_ids = tensor.matrix('network/target_word_ids',
                                             dtype='int64')
        if self.mode.minibatch:
            self.target_word_ids.tag.test_value = test_value(
                size=(20, 4), high=vocabulary.num_shortlist_words())
        else:
            self.target_word_ids.tag.test_value = test_value(
                size=(1, 4), high=vocabulary.num_shortlist_words())

        # mask is used to mask out the rest of the input matrix, when a sequence
        # is shorter than the maximum sequence length. The mask is kept as int8
        # data type, which is how Tensor stores booleans.
        if self.mode.minibatch:
            self.mask = tensor.matrix('network/mask', dtype='int8')
            self.mask.tag.test_value = test_value(size=(20, 4), high=True)
        else:
            self.mask = tensor.ones(self.input_word_ids.shape, dtype='int8')

        # Dropout layer needs to know whether we are training or evaluating.
        self.is_training = tensor.scalar('network/is_training', dtype='int8')
        self.is_training.tag.test_value = 1

        # num_noise_samples tells sampling based methods how many noise classes
        # to sample.
        self.num_noise_samples = tensor.scalar('network/num_noise_samples',
                                               dtype='int64')
        self.num_noise_samples.tag.test_value = 3
        self.noise_distribution = None

        for layer in self.layers.values():
            layer.create_structure()

    @classmethod
    def from_file(cls, model_path, mode=None, exclude_unk=False):
        """Reads a model from an HDF5 file.

        :type model_path: str
        :param model_path: path to a HDF5 model file

        :type mode: Network.Mode
        :param mode: selects mini-batch or single time step processing

        :type exclude_unk: bool
        :param exclude_unk: if set to ``True``, set ``<unk>`` probability to
                            zero before normalizing the network outputs
                            (required to get exact normalization during
                            inference)
        """

        with h5py.File(model_path, 'r') as state:
            print("Reading vocabulary from network state.")
            sys.stdout.flush()
            vocabulary = Vocabulary.from_state(state)
            print("Number of words in vocabulary:", vocabulary.num_words())
            print("Number of words in shortlist:", vocabulary.num_shortlist_words())
            print("Number of word classes:", vocabulary.num_classes())
            print("Building neural network.")
            sys.stdout.flush()
            architecture = Architecture.from_state(state)
            result = cls(architecture, vocabulary, mode=mode, exclude_unk=exclude_unk)
            print("Restoring neural network state.")
            sys.stdout.flush()
            result.set_state(state)
            return result

    def set_sampling(self, type, dampening, sharing):
        """Defines sampling of noise words for sampling-based output.

        When using a sampling based output layer, it needs to know the prior
        distribution of the classes, and how many words to sample. The number of
        noise classes per training word is controlled by the
        ``num_noise_samples`` tensor variable. The type of the prior
        distribution is defined by the ``type`` argument. If it is 'unigram',
        the unigram distribution of the classes in the training data is raised
        to the power of ``dampening`` and saved in a shared variable, so that we
        don't have to transfer the vector on every call of a Theano function.

        :type type: str
        :param type: either 'uniform', 'log-uniform' (logarithm of the samples
                     is uniformly distributed), or 'unigram' (unigram
                     distribution of the words raised to the power of
                     ``dampening``)

        :type dampening: float
        :param dampening: exponent to which unigram distribution is raised,
                          effectively dampening it

        :type sharing: str
        :param sharing: either ``None`` for k samples per mini-batch element,
                        'seq' for k samples per time step, or 'batch' for k
                        samples in total
        """

        output_layer = self.output_layer

        if (type == 'uniform') or ((type == 'unigram') and (dampening == 0.0)):
            distribution = UniformDistribution(self.random,
                                               self.vocabulary.num_classes())
        elif type == 'log-uniform':
            distribution = LogUniformDistribution(self.random,
                                                  self.vocabulary.num_classes())
        elif type == 'unigram':
            probs = numpy.power(self.class_prior_probs, dampening)
            probs /= probs.sum()
            probs = probs.astype(theano.config.floatX)
            if self._default_device is None:
                probs = theano.shared(probs, 'network/noise_probs')
            else:
                probs = theano.shared(probs, 'network/noise_probs',
                                      target=self._default_device)
            distribution = MultinomialDistribution(self.random, probs)
        else:
            raise ValueError("Invalid noise distribution requested: `{}'"
                             .format(type))

        if sharing is None:
            self._noise_sample, self._noise_sample_logprobs = \
                output_layer.get_sample_tensors(distribution)
        elif sharing == 'seq':
            self._noise_sample, self._noise_sample_logprobs = \
                output_layer.get_seqshared_sample_tensors(distribution)
        elif sharing == 'batch':
            self._noise_sample, self._noise_sample_logprobs = \
                output_layer.get_shared_sample_tensors(distribution)
        else:
            raise ValueError("Unknown noise sample sharing: `{}'"
                             .format(sharing))

        self.noise_distribution = distribution

    def get_state(self, state):
        """Pulls parameter values from Theano shared variables.

        If there already is a parameter in the state, it will be replaced, so it
        has to have the same number of elements.

        :type state: h5py.File
        :param state: HDF5 file for storing the neural network parameters
        """

        for layer in self.layers.values():
            layer.get_state(state)

        self.architecture.get_state(state)

    def set_state(self, state):
        """Sets the values of Theano shared variables.

        Requires that ``state`` contains values for all the neural network
        parameters.

        :type state: h5py.File
        :param state: HDF5 file that contains the neural network parameters
        """

        for layer in self.layers.values():
            layer.set_state(state)

        try:
            self.architecture.check_state(state)
        except IncompatibleStateError as error:
            raise IncompatibleStateError(
                "Attempting to restore state of a network that is incompatible "
                "with this architecture. " + str(error))

    def get_variables(self):
        """Returns a dictionary of the shared variables.

        This function is used by the optimizers to create optimization
        parameters that are specific to network parameters, and compute
        gradients with regard to the parameters.

        :rtype: dict
        :returns: mapping from parameter path to Theano shared variables
        """

        result = dict()
        for layer in self.layers.values():
            result.update(layer.get_variables())
        return result

    def add_recurrent_state(self, size):
        """Adds a recurrent state variable and returns its index.

        Used by recurrent layers to add a state variable that has to be passed
        from one time step to the next, when generating text or computing
        lattice probabilities.

        :type size: int
        :param size: size of the state vector

        :rtype size: int
        :param size: index of the new recurrent state variable
        """

        index = len(self.recurrent_state_size)
        assert index == len(self.recurrent_state_input)

        # The variables are in the structure of a mini-batch (3-dimensional
        # array) to keep the layer functions general.
        variable = tensor.tensor3('network/recurrent_state_' + str(index),
                                  dtype=theano.config.floatX)
        variable.tag.test_value = test_value(size=(1, 4, size), high=1.0)

        self.recurrent_state_size.append(size)
        self.recurrent_state_input.append(variable)

        return index

    def output_probs(self):
        """Returns the output probabilities for the whole vocabulary.

        :rtype: Variable
        :returns: a symbolic 3-dimensional matrix that contains a probability
                  for each time step, each sequence, and each output class
        """

        if not hasattr(self.output_layer, 'output_probs'):
            raise RuntimeError("The final layer is not an output layer.")
        return self.output_layer.output_probs

    def target_probs(self):
        """Returns the output probabilities for the predicted words.

        Can be used only when target_class_ids is given.

        :rtype: Variable
        :returns: a symbolic 2-dimensional matrix that contains the target word
                  probability for each time step and each sequence
        """

        if not hasattr(self.output_layer, 'target_probs'):
            raise RuntimeError("The final layer is not an output layer.")
        return self.output_layer.target_probs

    def unnormalized_logprobs(self):
        """Returns the unnormalized log probabilities for the predicted words.

        These are the preactivations of the output layer, before softmax. As the
        softmax output is exponential, these can be seen as the unnormalized log
        probabilities.

        Only computed when target_class_ids is given and using softmax output.

        :rtype: Variable
        :returns: a symbolic 2-dimensional matrix that contains the unnormalized
                  target word probability for each time step and each sequence
        """

        if not hasattr(self.output_layer, 'unnormalized_logprobs'):
            raise RuntimeError("The final layer is not a softmax layer, and "
                               "unnormalized probabilities are needed.")
        return self.output_layer.unnormalized_logprobs

    def noise_sample(self):
        """Returns the classes sampled from a noise distribution, and their log
        probabilities.

        Only computed when target_class_ids is given and using softmax output.

        If ``sharing`` is None``, the sample is a 3-dimensional matrix of k
        class IDs for each time step and sequence. If ``sharing`` is 'seq', the
        sample is a 2-dimensional matrix, k class IDs for each time step. If
        ``sharing`` is 'batch', the sample is a vector of k class IDs in total.
        The log probabilities are always returned in a 3-dimensional matrix, as
        they differ for each time step and sequence.

        :rtype: tuple of two Variables
        :returns: noise class IDs and their log probabilities
        """

        if (self._noise_sample is None) or \
           (self._noise_sample_logprobs is None):
            raise RuntimeError(
                "Trying to read the noise sample before defining sampling.")

        return self._noise_sample, self._noise_sample_logprobs

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
                                     "creating layer `{}'."
                                     .format(e.args[0], description['name']))
            else:
                result[variable] = value
        return result
