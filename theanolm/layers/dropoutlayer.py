#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import logging
import numpy
import theano
import theano.tensor as tensor
from theanolm.layers.basiclayer import BasicLayer

class DropoutLayer(BasicLayer):
    """Dropout Layer for Neural Network Language Model

    A dropout layer is not a regular layer in the sense that it doesn't contain
    any neurons. It simply randomly sets some activations to zero at train time
    to prevent overfitting.    

    N. Srivastava et al. (2014)
    Dropout: A Simple Way to Prevent Neural Networks from Overfitting
    Journal of Machine Learning Research, 15, 1929-1958
    """

    def __init__(self, layer_options, *args, **kwargs):
        """Validates the parameters of this layer.
        """

        super().__init__(layer_options, *args, **kwargs)

        if 'dropout_rate' in layer_options:
            self.dropout_rate = float(layer_options['dropout_rate'])
            # Don't allow dropout rate too close to 1.0 to prevent division by
            # zero.
            self.dropout_rate = min(0.9999, self.dropout_rate)
        else:
            self.dropout_rate = 0.5
        logging.debug("  dropout_rate=%f", self.dropout_rate)

        # Make sure the user hasn't tried to change the number of connections.
        input_size = self.input_layers[0].output_size
        output_size = self.output_size
        if input_size != output_size:
            raise ValueError("Dropout layer cannot change the number of connections.")

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        Sets self.output to a symbolic matrix that describes the output of this
        layer. When training, masks the output Assumes that the shared variables have been passed using
        ``set_params()``.
        """

        layer_input = self.input_layers[0].output
        # Pass rate is the probability of not dropping a unit.
        pass_rate = 1.0 - self.dropout_rate
        mask = self.network.random.binomial(
            layer_input.shape, p=pass_rate, n=1, dtype=layer_input.dtype)
        # Multiply the output by the inverse of the pass rate before dropping
        # units to compensate the scaling effect.
        scale_correction = 1.0 / pass_rate
        self.output = tensor.switch(
            self.network.is_training,
            layer_input * scale_correction * tensor.cast(mask, theano.config.floatX),
            layer_input)
