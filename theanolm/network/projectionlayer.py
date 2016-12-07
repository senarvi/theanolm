#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.network.basiclayer import BasicLayer

class ProjectionLayer(BasicLayer):
    """Projection Layer

    Projection layer supports dividing the weight to multiple devices. The
    second dimension of the projection matrix (output space) is split to equal
    parts. After projecting on each part, the output will be concatenated.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the parameters used by this layer.
        """

        super().__init__(*args, **kwargs)

        # Initialize the parameters.
        input_size = sum(x.output_size for x in self.input_layers)
        output_size = self.output_size
        self._init_weight('W', (input_size, output_size), scale=0.01,
                          split_to_devices=True)

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        The input is always 2-dimensional: the first dimension is the time step
        (index of word in a sequence) and the second dimension are the
        sequences.

        Sets self.output to a symbolic matrix that describes the output of this
        layer.
        """

        assert len(self.input_layers) == 1
        layer_input = self.input_layers[0].output
        num_time_steps = layer_input.shape[0]
        num_sequences = layer_input.shape[1]

        outputs = []
        for device in self._devices:
            # Indexing the word_projection matrix with a word ID returns the
            # self.output_size dimensional projection. Note that indexing the
            # matrix with a vector of all the word IDs gives a concatenation of
            # those projections.
            device_output = self._get_param('W', device)[layer_input.flatten()]
            device_output = device_output.reshape([num_time_steps,
                                                   num_sequences,
                                                   -1])
            outputs.append(device_output)
        self.output = tensor.concatenate(outputs, axis=2)
