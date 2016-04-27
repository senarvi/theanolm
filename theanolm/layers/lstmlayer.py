#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as tensor
from theanolm.matrixfunctions import get_submatrix
from theanolm.layers.basiclayer import BasicLayer

class LSTMLayer(BasicLayer):
    """Long Short-Term Memory Layer for Neural Network Language Model

    A. Graves, J. Schmidhuber (2005)
    Framewise phoneme classification with bidirectional LSTM and other neural
    network architectures
    Neural Networks, 18(5–6), 602–610
    """

    def __init__(self, *args, **kwargs):
        """Initializes the parameters used by this layer.

        The weight matrices are concatenated so that they can be applied in a
        single parallel matrix operation. The same thing for bias vectors.
        Input, forget, and output gate biases are initialized to -1.0, 1.0, and
        -1.0 respectively, so that in the beginning of training, the forget gate
        activation will be almost 1.0 (meaning that the LSTM does not default to
        forgetting information).
        """

        super().__init__(*args, **kwargs)

        input_size = self.input_layers[0].output_size
        output_size = self.output_size

        # The number of state variables to be passed between time steps.
        self.cell_state_index = self.network.add_recurrent_state(output_size)
        self.hidden_state_index = self.network.add_recurrent_state(output_size)

        # Initialize the parameters.
        num_gates = 3
        # layer input weights for each gate and the candidate state
        self._init_orthogonal_weight('layer_input/W', input_size, output_size,
                                     scale=0.01, count=num_gates+1)
        # hidden state input weights for each gate and the candidate state
        self._init_orthogonal_weight('step_input/W', output_size, output_size,
                                     count=num_gates+1)
        # biases for each gate and the candidate state
        self._init_bias('layer_input/b', output_size, [-1.0, 1.0, -1.0, 0.0])

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        The input is always 3-dimensional: the first dimension is the time step,
        the second dimension are the sequences, and the third dimension is the
        layer input. If ``self.network.batch_processing`` is ``True``, the
        function creates the normal mini-batch structure.

        The function can also be used to create a structure for generating text,
        one word at a time. Then the input is still 3-dimensional, but the size
        of the first and second dimension is 1, and the state outputs from the
        previous time step are read from ``self.network.recurrent_state_input``.

        Saves the recurrent state in the Network object: cell state C_(t) and
        hidden state h_(t). ``self.output`` will be set to the hidden state
        output, which is the actual output of this layer.

        Assumes that the shared variables have been passed using
        ``set_params()``.
        """

        input_matrix = self.input_layers[0].output
        num_time_steps = input_matrix.shape[0]
        num_sequences = input_matrix.shape[1]

        # Compute the gate and candidate state pre-activations, which don't
        # depend on the state input from the previous time step.
        layer_input_preact = self._tensor_preact(input_matrix, 'layer_input')

        # Weights of the hidden state input of each time step have to be applied
        # inside the loop.
        hidden_state_weights = self._get_param('step_input/W')

        if self.network.batch_processing:
            sequences = [self.network.mask, layer_input_preact]
            non_sequences = [hidden_state_weights]
            initial_value = numpy.dtype(theano.config.floatX).type(0.0)
            initial_cell_state = \
                tensor.alloc(initial_value, num_sequences, self.output_size)
            initial_hidden_state = \
                tensor.alloc(initial_value, num_sequences, self.output_size)

            state_outputs, _ = theano.scan(
                self._create_time_step,
                sequences=sequences,
                outputs_info=[initial_cell_state, initial_hidden_state],
                non_sequences=non_sequences,
                name='hidden_layer_steps',
                n_steps=num_time_steps,
                profile=self._profile,
                strict=True)
            self.network.recurrent_state_output[self.cell_state_index] = \
                state_outputs[0]
            self.network.recurrent_state_output[self.hidden_state_index] = \
                state_outputs[1]
        else:
            cell_state_input = \
                self.network.recurrent_state_input[self.cell_state_index]
            hidden_state_input = \
                self.network.recurrent_state_input[self.hidden_state_index]

            state_outputs = self._create_time_step(
                self.network.mask,
                layer_input_preact,
                cell_state_input,
                hidden_state_input,
                hidden_state_weights)
            self.network.recurrent_state_output[self.cell_state_index] = \
                state_outputs[0]
            self.network.recurrent_state_output[self.hidden_state_index] = \
                state_outputs[1]

        self.output = state_outputs[1]

    def _create_time_step(self, mask, x_preact, C_in, h_in, h_weights):
        """The LSTM step function for theano.scan(). Creates the structure of
        one time step.

        The inputs ``mask`` and ``x_preact`` contain only one time step, but
        possibly multiple sequences. There may, or may not be the first
        dimension of size 1 - it won't affect the computations, because
        broadcasting works by aligning the last dimensions.

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
                         biases

        :type C_in: TensorVariable
        :param C_in: C_(t-1), cell state output from the previous time step

        :type h_in: TensorVariable
        :param h_in: h_(t-1), hidden state output of the previous time step

        :type h_weights: TensorVariable
        :param h_weights: concatenation of the gate and candidate state weights
                          to be applied to h_(t-1)

        :rtype: a tuple of two TensorVariables
        :returns: C_(t) and h_(t), the cell state and hidden state outputs
        """

        # pre-activation of the gates and candidate state
        preact = tensor.dot(h_in, h_weights)
        preact += x_preact

        # input, forget, and output gates
        i = tensor.nnet.sigmoid(get_submatrix(preact, 0, self.output_size))
        f = tensor.nnet.sigmoid(get_submatrix(preact, 1, self.output_size))
        o = tensor.nnet.sigmoid(get_submatrix(preact, 2, self.output_size))

        # cell state and hidden state outputs
        C_candidate = tensor.tanh(get_submatrix(preact, 3, self.output_size))
        C_out = f * C_in + i * C_candidate
        h_out = o * tensor.tanh(C_out)

        # Apply the mask. None creates a new axis with size 1, causing the mask
        # to be broadcast to all the outputs.
        C_out = tensor.switch(mask[:,None], C_out, C_in)
        h_out = tensor.switch(mask[:,None], h_out, h_in)

        return C_out, h_out
