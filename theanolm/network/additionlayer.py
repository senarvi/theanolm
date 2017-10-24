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

    def __init__(self, layer_options, *args, **kwargs):
        """If the input dimensionality does not change, this layer has no
        parameters. A weight matrix for matching the dimensions is created for
        every input that has different dimensionality than the output.

        If the output size is not specified, the size of the first input will be
        assumed.

        :type layer_options: dict
        :param layer_options: dictionary of layer options
        """

        if 'size' not in layer_options:
            layer_options['size'] = layer_options['input_layers'][0].output_size

        super().__init__(layer_options, *args, **kwargs)

        for input_index, input_layer in enumerate(self._input_layers):
            input_size = input_layer.output_size
            if input_size != self.output_size:
                param_name = 'input{}/W'.format(input_index)
                self._init_weight(param_name, (input_size, self.output_size),
                                  scale=0.01)

        self.output = None

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        Sets self.output to a symbolic matrix that describes the output of this
        layer. If the inputs are the same size as the output, the output will be
        the elementwise sum of the inputs. If needed, the inputs will be
        projected to the same size.
        """

        for input_index, input_layer in enumerate(self._input_layers):
            input_size = input_layer.output_size
            if input_size == self.output_size:
                input_matrix = input_layer.output
            else:
                param_name = 'input{}/W'.format(input_index)
                weight = self._params[self._param_path(param_name)]
                input_matrix = tensor.dot(input_layer.output, weight)

            if self.output is None:
                self.output = input_matrix
            else:
                self.output = tensor.add(self.output, input_matrix)
