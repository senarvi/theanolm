#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the addition layer.
"""

import theano.tensor as tensor

from theanolm.network.basiclayer import BasicLayer

class AdditionLayer(BasicLayer):
    """Addition Layer

    A layer that produces an elementwise sum of its inputs. All inputs have to
    be equal size.
    """

    def __init__(self, *args, **kwargs):
        """This layer has no parameters. Just verifies that all inputs are equal
        size.
        """

        super().__init__(*args, **kwargs)

        input_size = self._input_layers[0].output_size
        for input_layer in self._input_layers[1:]:
            if input_layer.output_size != input_size:
               raise ValueError("All inputs of an addition layer have to be "
                                "equal size.")

        self.output = None

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        Sets self.output to a symbolic matrix that describes the output of this
        layer, which is the sum of its inputs.
        """

        self.output = self._input_layers[0].output
        for input_layer in self._input_layers[1:]:
            self.output = tensor.add(self.output, input_layer.output)
