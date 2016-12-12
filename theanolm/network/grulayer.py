#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.network.weightfunctions import get_submatrix
from theanolm.network.basiclayer import BasicLayer

class GRULayer(BasicLayer):
    """Gated Recurrent Unit Layer

    K. Cho et al. (2014)
    Learning Phrase Representations Using RNN Encoder-Decoder for Statistical
    Machine Translation
    Proc. 2014 Conference on Empiricial Methods in Natural Language Processing
    """

    def __init__(self, *args, **kwargs):
        """Initializes the parameters used by this layer.

        The weight matrices are concatenated so that they can be applied in a
        single parallel matrix operation. The same thing for bias vectors.
        """

        super().__init__(*args, **kwargs)

        input_size = sum(x.output_size for x in self.input_layers)
        output_size = self.output_size

        # Add state variables to be passed between time steps.
        self.hidden_state_index = self._network.add_recurrent_state(output_size)

        # Initialize the parameters.
        num_gates = 2
        # layer input weights for each gate and the candidate state
        self._init_weight('layer_input/W', (input_size, output_size),
                          scale=0.01, count=num_gates+1)
        # hidden state input weights for each gate and the candidate state
        self._init_weight('step_input/W', (output_size, output_size),
                          count=num_gates+1)
        # biases for each gate and the candidate state
        self._init_bias('layer_input/b', output_size, [None] * (num_gates + 1))

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        The input is always 3-dimensional: the first dimension is the time step,
        the second dimension are the sequences, and the third dimension is the
        layer input. When processing mini-batches, all dimensions can have size
        greater than one.

        The function can also be used to create a structure for generating the
        probability distribution of the next word. Then the input is still
        3-dimensional, but the size of the first dimension (time steps) is 1,
        and the state outputs from the previous time step are read from
        ``self._network.recurrent_state_input``.

        Saves the recurrent state in the Network object. There's just one state
        in a GRU layer, h_(t). ``self.output`` will be set to the same hidden
        state output, which is also the actual output of this layer.
        """

        layer_input = tensor.concatenate([x.output for x in self.input_layers],
                                         axis=2)
        num_time_steps = layer_input.shape[0]
        num_sequences = layer_input.shape[1]

        # Compute the gate and candidate state pre-activations, which don't
        # depend on the state input from the previous time step.
        layer_input_preact = self._tensor_preact(layer_input, 'layer_input')

        # Weights of the hidden state input of each time step have to be applied
        # inside the loop.
        hidden_state_weights = self._get_param('step_input/W')

        if self._network.mode.minibatch:
            sequences = [self._network.mask, layer_input_preact]
            non_sequences = [hidden_state_weights]
            initial_hidden_state = tensor.zeros(
                (num_sequences, self.output_size), dtype=theano.config.floatX)

            hidden_state_output, _ = theano.scan(
                self._create_time_step,
                sequences=sequences,
                outputs_info=[initial_hidden_state],
                non_sequences=non_sequences,
                name='gru_steps',
                n_steps=num_time_steps,
                profile=self._profile,
                strict=True)

            self.output = hidden_state_output
        else:
            hidden_state_input = \
                self._network.recurrent_state_input[self.hidden_state_index]

            hidden_state_output = self._create_time_step(
                self._network.mask[0],
                layer_input_preact[0],
                hidden_state_input[0],
                hidden_state_weights)

            # Create a new axis for time step with size 1.
            hidden_state_output = hidden_state_output[None,:,:]

            self._network.recurrent_state_output[self.hidden_state_index] = \
                hidden_state_output
            self.output = hidden_state_output

    def _create_time_step(self, mask, x_preact, h_in, h_weights):
        """The GRU step function for theano.scan(). Creates the structure of one
        time step.

        The inputs do not contain the time step dimension. ``mask`` is a vector
        containing a boolean mask for each sequence. ``x_preact`` is a matrix
        containing the preactivations for each sequence. ``C_in`` and ``h_in``,
        as well as the outputs, are matrices containing the state vectors for
        each sequence.

        The required affine transformations have already been applied to the
        input prior to creating the loop. The transformed inputs and the mask
        that will be passed to the step function are vectors when processing a
        mini-batch - each value corresponds to the same time step in a different
        sequence.

        :type mask: TensorVariable
        :param mask: a symbolic vector that masks out sequences that are past
                     the last word

        :type x_preact: TensorVariable
        :param x_preact: concatenation of the input x_(t) pre-activations
                         computed using the gate and candidate state weights and
                         biases; shape is (the number of sequences, state size *
                         3)

        :type h_in: TensorVariable
        :param h_in: h_(t-1), hidden state output of the previous time step;
                     shape is (the number of sequences, state size)

        :type h_weights: TensorVariable
        :param h_weights: concatenation of the gate and candidate state weights
                          to be applied to h_(t-1); shape is (state size, state
                          size * 3)

        :rtype: TensorVariable
        :returns: h_(t), the hidden state output
        """

        # pre-activation of the gates
        h_preact = tensor.dot(h_in, h_weights)
        preact_gates = get_submatrix(h_preact, 0, self.output_size, 1)
        preact_gates += get_submatrix(x_preact, 0, self.output_size, 1)

        # reset and update gates
        r = tensor.nnet.sigmoid(get_submatrix(preact_gates, 0, self.output_size))
        u = tensor.nnet.sigmoid(get_submatrix(preact_gates, 1, self.output_size))

        # pre-activation of the candidate state
        preact_candidate = get_submatrix(h_preact, 2, self.output_size)
        preact_candidate *= r
        preact_candidate += get_submatrix(x_preact, 2, self.output_size)

        # hidden state output
        h_candidate = tensor.tanh(preact_candidate)
        h_out = (1.0 - u) * h_in + u * h_candidate

        # Apply the mask. None creates a new axis with size 1, causing the mask
        # to be broadcast to all the outputs.
        h_out = tensor.switch(mask[:,None], h_out, h_in)

        return h_out
