#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.matrixfunctions import get_submatrix
from theanolm.layers.basiclayer import BasicLayer

class LSTMLayer(BasicLayer):
    """Long Short-Term Memory Layer for Neural Network Language Model
    """

    def __init__(self, layer_name, input_layers, output_size, profile):
        """Initializes the parameters for this layer.

        :type layer_name: str
        :param layer_name: name of the layer, used for prefixing parameter names

        :type input_layer: list of BasicLayers
        :param input_layer: list of layers providing input to this layer

        :type output_size: int
        :param output_size: number of output connections

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        super().__init__(layer_name, input_layers, output_size, is_recurrent=True)
        self._profile = profile

        # The number of state variables to be passed between time steps.
        self.num_state_variables = 2

        # Initialize the parameters.
        input_size = self.input_layers[0].output_size
        num_gates = 3

        # concatenation of the input weights for each gate
        self._init_orthogonal_weight('gates.W', input_size, output_size, scale=0.01, count=num_gates)
        # concatenation of the previous step output weights for each gate
        self._init_orthogonal_weight('gates.U', output_size, output_size, count=num_gates)
        # concatenation of the biases for each gate
        self._init_zero_bias('gates.b', num_gates * output_size)

        # input weight for the candidate state
        self._init_orthogonal_weight('candidate.W', input_size, output_size, scale=0.01)
        # previous step output weight for the candidate state
        self._init_orthogonal_weight('candidate.U', output_size, output_size)
        # bias for the candidate state
        self._init_zero_bias('candidate.b', output_size)

    def create_structure(self, mask, state_inputs=None):
        """Creates the symbolic graph of this layer.

        The input is always 3-dimensional: the first dimension is the time step,
        the second dimension are the sequences, and the third dimension is the
        layer input. If ``state_inputs`` is ``None``, the function creates the
        normal recursive structure.

        The function can also be used to create a structure for generating text,
        one word at a time. Then the input is still 3-dimensional, but the size
        of the first and second dimension is 1, and the state outputs from the
        previous time step are provided in ``state_inputs``.

        Sets ``self.state_outputs`` to a list of symbolic 3-dimensional matrices
        that describe the state outputs: cell state C_(t) and hidden state
        h_(t). ``self.output`` will be set to the hidden state output, which is
        the actual output of this layer.

        Assumes that the shared variables have been passed using
        ``set_params()``.

        :type mask: theano.tensor.var.TensorVariable
        :param mask: symbolic 2-dimensional matrix that masks out time steps in
                     layer_input after sequence end

        :type state_inputs: list of theano.tensor.var.TensorVariables
        :param state_inputs: a list of symbolic 3-dimensional matrices that
                            describe the state outputs of the previous time step
                            - cell state C_(t-1) and hidden state h_(t-1)
        """

        input_matrix = self.input_layers[0].output
        num_time_steps = input_matrix.shape[0]
        num_sequences = input_matrix.shape[1]
        self.layer_size = self._get_param('candidate.U').shape[1]

        # Compute the gate pre-activations, which don't depend on the time step.
        x_preact_gates = self._tensor_preact(input_matrix, 'gates')
        x_preact_candidate = self._tensor_preact(input_matrix, 'candidate')

        # The weights for the previous step output. These have to be applied
        # inside the loop.
        U_gates = self._get_param('gates.U')
        U_candidate = self._get_param('candidate.U')

        if state_inputs is None:
            sequences = [mask, x_preact_gates, x_preact_candidate]
            non_sequences = [U_gates, U_candidate]
            initial_value = numpy.dtype(theano.config.floatX).type(0.0)
            initial_cell_state = tensor.unbroadcast(
                tensor.alloc(initial_value, num_sequences, self.layer_size), 0)
            initial_hidden_state = tensor.unbroadcast(
                tensor.alloc(initial_value, num_sequences, self.layer_size), 0)

            self.state_outputs, _ = theano.scan(
                self._create_time_step,
                sequences=sequences,
                outputs_info=[initial_cell_state, initial_hidden_state],
                non_sequences=non_sequences,
                name='hidden_layer_steps',
                n_steps=num_time_steps,
                profile=self._profile,
                strict=True)
        else:
            cell_state_input = state_inputs[0]
            hidden_state_input = state_inputs[1]

            self.state_outputs = self._create_time_step(
                mask,
                x_preact_gates,
                x_preact_candidate,
                cell_state_input,
                hidden_state_input,
                U_gates,
                U_candidate)

        self.output = self.state_outputs[1]

    def _create_time_step(self, mask, x_preact_gates, x_preact_candidate, C_in,
                          h_in, U_gates, U_candidate):
        """The LSTM step function for theano.scan(). Creates the structure of
        one time step.

        The required affine transformations have already been applied to the
        input prior to creating the loop. The transformed inputs and the mask
        that will be passed to the step function are vectors when processing a
        mini-batch - each value corresponds to the same time step in a different
        sequence.

        :type mask: theano.tensor.var.TensorVariable
        :param mask: masks out time steps after sequence end

        :type x_preact_gates: theano.tensor.var.TensorVariable
        :param x_preact_gates: concatenation of the input x_(t) pre-activations
                               computed using the various gate weights and
                               biases

        :type x_preact_candidate: theano.tensor.var.TensorVariable
        :param x_preact_candidate: input x_(t) pre-activation computed using the
                                   weight W and bias b for the new candidate
                                   state

        :type C_in: theano.tensor.var.TensorVariable
        :param C_in: C_(t-1), cell state output from the previous time step

        :type h_in: theano.tensor.var.TensorVariable
        :param h_in: h_(t-1), hidden state output of the previous time step

        :type U_gates: theano.tensor.var.TensorVariable
        :param U_gates: concatenation of the gate weights to be applied to
                        h_(t-1)

        :type U_candidate: theano.tensor.var.TensorVariable
        :param U_candidate: candidate state weight matrix to be applied to
                            h_(t-1)

        :rtype: a tuple of two theano.tensor.var.TensorVariables
        :returns: C_(t) and h_(t), the cell state and hidden state outputs
        """

        # pre-activation of the gates
        preact_gates = tensor.dot(h_in, U_gates)
        preact_gates += x_preact_gates

        # input, forget, and output gates
        i = tensor.nnet.sigmoid(get_submatrix(preact_gates, 0, self.layer_size))
        f = tensor.nnet.sigmoid(get_submatrix(preact_gates, 1, self.layer_size))
        o = tensor.nnet.sigmoid(get_submatrix(preact_gates, 2, self.layer_size))

        # pre-activation of the candidate state
        preact_candidate = tensor.dot(h_in, U_candidate)
        preact_candidate += x_preact_candidate

        # cell state and hidden state outputs
        C_candidate = tensor.tanh(preact_candidate)
        C_out = f * C_in + i * C_candidate
        h_out = o * tensor.tanh(C_out)

        # Apply the mask.
        C_out = mask[:, None] * C_out + (1.0 - mask)[:, None] * C_in
        h_out = mask[:, None] * h_out + (1.0 - mask)[:, None] * h_in

        return C_out, h_out
