#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import logging
import numpy
import theano
import theano.tensor as tensor
from theanolm.network.basiclayer import BasicLayer

class HSoftmaxLayer(BasicLayer):
    """Hierarchical Softmax Output Layer

    The output layer is a simple softmax layer that outputs the word
    probabilities.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the parameters used by this layer.
        """

        super().__init__(*args, **kwargs)

        # Factorize the output size into two levels of equal or almost equal
        # size.
        output_size = self.output_size
        level1_size = numpy.int(numpy.ceil(numpy.sqrt(output_size)))
        level2_size = numpy.int(numpy.ceil(output_size / level1_size))
        assert level1_size * level2_size >= output_size
        assert (level1_size - 1) * level2_size < output_size
        assert level1_size * (level2_size - 1) < output_size
        assert level1_size == level2_size or level1_size == level2_size + 1
        self.level1_size = level1_size
        self.level2_size = level2_size
        logging.debug("  level1_size=%d level2_size=%d",
                      self.level1_size,
                      self.level2_size)

        # Create the parameters. Weight matrix and bias for concatenated input
        # and the second level of the hierarchy.
        input_size = sum(x.output_size for x in self.input_layers)
        self._init_random_weight('input/W',
                                 (input_size, level1_size),
                                 scale=0.01)
        self._init_bias('input/b', level1_size)
        self._init_random_weight('level1/W',
                                 (level1_size, input_size, level2_size),
                                 scale=0.01)
        self._init_bias('level1/b', (level1_size, level2_size))

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        The input is always 3-dimensional: the first dimension is the time step,
        the second dimension are the sequences, and the third dimension is the
        word projection. When generating text, there's just one sequence and one
        time step in the input.

        Sets self.output to a symbolic matrix that describes the output of this
        layer. Assumes that the shared variables have been passed using
        ``set_params()``.
        """

        layer_input = tensor.concatenate([x.output for x in self.input_layers],
                                         axis=2)

        # If we're only predicting probabilities of the target outputs, the
        # targets are the words at the next time step and the last time step is
        # not used as input.
        if self.network.predict_next_distribution:
            target_class_ids = None
        else:
            layer_input = layer_input[:-1]
            target_class_ids = self.network.class_input[1:].flatten()

        # Combine the first two dimensions so that softmax is taken
        # independently for each location, over the output classes.
        num_time_steps = layer_input.shape[0]
        num_sequences = layer_input.shape[1]
        input_size = layer_input.shape[2]
        layer_input = layer_input.reshape([num_time_steps * num_sequences,
                                          input_size])

        input_weight = self._params[self._param_path('input/W')]
        input_bias = self._params[self._param_path('input/b')]
        level1_weight = self._params[self._param_path('level1/W')]
        level1_bias = self._params[self._param_path('level1/b')]

        probs = tensor.nnet.h_softmax(layer_input,
                                      num_time_steps * num_sequences,
                                      self.output_size,
                                      self.level1_size,
                                      self.level2_size,
                                      input_weight,
                                      input_bias,
                                      level1_weight,
                                      level1_bias,
                                      target_class_ids)

        if self.network.predict_next_distribution:
            self.output_probs = probs.reshape([num_time_steps,
                                               num_sequences,
                                               self.output_size])
            self.target_probs = None
        else:
            self.output_probs = None
            self.target_probs = probs.reshape([num_time_steps,
                                               num_sequences])
