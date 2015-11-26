#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.matrixfunctions import orthogonal_weight, get_submatrix

class LSTMLayer(object):
    """Long Short-Term Memory Layer for Neural Network Language Model
    """

    def __init__(self, in_size, out_size, profile=False):
        """Initializes the parameters for an LSTM layer of a recurrent neural
        network.

        :type in_size: int
        :param in_size: number of input connections

        :type out_size: int
        :param out_size: number of output connections

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        self._profile = profile

        # The number of state variables to be passed between time steps.
        self.num_state_variables = 2

        # Initialize the parameters.
        self.param_init_values = OrderedDict()

        num_gates = 3

        # concatenation of the input weights for each gate
        self.param_init_values['lstm.W_gates'] = \
            numpy.concatenate([orthogonal_weight(in_size, out_size, scale=0.01)
                               for _ in range(num_gates)],
                              axis=1)

        # concatenation of the previous step output weights for each gate
        self.param_init_values['lstm.U_gates'] = \
            numpy.concatenate([orthogonal_weight(out_size, out_size)
                               for _ in range(num_gates)],
                              axis=1)

        # concatenation of the biases for each gate
        self.param_init_values['lstm.b_gates'] = \
                numpy.zeros((num_gates * out_size,)).astype(theano.config.floatX)

        # input weight for the candidate state
        self.param_init_values['lstm.W_candidate'] = \
                orthogonal_weight(in_size, out_size, scale=0.01)

        # previous step output weight for the candidate state
        self.param_init_values['lstm.U_candidate'] = \
                orthogonal_weight(out_size, out_size)

        # bias for the candidate state
        self.param_init_values['lstm.b_candidate'] = \
                numpy.zeros((out_size,)).astype(theano.config.floatX)

    def create_structure(self, model_params, layer_input, mask, state_inputs=None):
        """Creates LSTM layer structure.

        The input is always 3-dimensional: the first dimension is the time step,
        the second dimension are the sequences, and the third dimension is the
        layer input. If ``state_inputs`` is ``None``, the function creates the
        normal recursive structure.

        The function can also be used to create a structure for generating text,
        one word at a time. Then the input is still 3-dimensional, but the size
        of the first and second dimension is 1, and the state outputs from the
        previous time step are provided in ``state_inputs``.

        Sets ``self.output`` to a symbolic 2-dimensional matrix that describes
        the output of this layer. If ``state_inputs`` is given, sets also
        ``self.state_output`` to a list of symbolic 2-dimensional matrices that
        describe all the state outputs: cell state C_(t) and hidden state h_(t).

        :type model_params: dict
        :param model_params: shared Theano variables

        :type layer_input: theano.tensor.var.TensorVariable
        :param layer_input: x_(t), symbolic 3-dimensional matrix that describes
                            the output of the previous layer (word projections
                            of the sequences)

        :type mask: theano.tensor.var.TensorVariable
        :param mask: symbolic 2-dimensional matrix that masks out time steps in
                     layer_input after sequence end

        :type state_inputs: list of theano.tensor.var.TensorVariables
        :param state_inputs: a list of symbolic 3-dimensional matrices that
                            describe the state outputs of the previous time step
                            - cell state C_(t-1) and hidden state h_(t-1)
        """

        num_time_steps = layer_input.shape[0]
        num_sequences = layer_input.shape[1]
        self.layer_size = model_params['lstm.U_candidate'].shape[1]

        # Compute the gate pre-activations, which don't depend on the time step.
        x_preact_gates = \
                tensor.dot(layer_input, model_params['lstm.W_gates']) \
                + model_params['lstm.b_gates']
        x_preact_candidate = \
                tensor.dot(layer_input, model_params['lstm.W_candidate']) \
                + model_params['lstm.b_candidate']

        # The weights and biases for the previous step output. These have to be
        # applied inside the loop.
        U_gates = model_params['lstm.U_gates']
        U_candidate = model_params['lstm.U_candidate']

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
        print("output dim =", self.output.ndim)

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
