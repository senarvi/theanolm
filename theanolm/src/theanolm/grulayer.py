#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.matrixfunctions import orthogonal_weight, get_submatrix

class GRULayer(object):
    """Gated Recurrent Unit Layer for Neural Network Language Model
    """

    def __init__(self, in_size, out_size, profile):
        """Initializes the parameters for a GRU layer of a recurrent neural
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
        self.num_state_variables = 1

        # Initialize the parameters.
        self.param_init_values = OrderedDict()

        num_gates = 2

        # concatenation of the input weights for each gate
        self.param_init_values['encoder_W_gates'] = \
                numpy.concatenate([orthogonal_weight(in_size, out_size, scale=0.01) for _ in range(num_gates)],
                                  axis=1)

        # concatenation of the previous step output weights for each gate
        self.param_init_values['encoder_U_gates'] = \
                numpy.concatenate([orthogonal_weight(out_size, out_size) for _ in range(num_gates)],
                                  axis=1)

        # concatenation of the biases for each gate
        self.param_init_values['encoder_b_gates'] = \
                numpy.zeros((num_gates * out_size,)).astype('float32')

        # input weight for the candidate state
        self.param_init_values['encoder_W_candidate'] = \
                orthogonal_weight(in_size, out_size, scale=0.01)

        # previous step output weight for the candidate state
        self.param_init_values['encoder_U_candidate'] = \
                orthogonal_weight(out_size, out_size)

        # bias for the candidate state
        self.param_init_values['encoder_b_candidate'] = \
                numpy.zeros((out_size,)).astype('float32')

    def create_minibatch_structure(self, model_params, layer_input, mask):
        """Creates GRU layer structure for mini-batch processing.

        In mini-batch training the input is 3-dimensional: the first
        dimension is the time step, the second dimension are the sequences,
        and the third dimension is the word projection.

        Sets self.minimatch_output to a symbolic 2-dimensional matrix that
        describes the hidden state output of the time steps.

        :type model_params: dict
        :param model_params: shared Theano variables

        :type layer_input: theano.tensor.var.TensorVariable
        :param layer_input: x_(t), symbolic 3-dimensional matrix that describes
                            the output of the previous layer (word projections
                            of the sequences)

        :type mask: theano.tensor.var.TensorVariable
        :param mask: symbolic 2-dimensional matrix that masks out time steps in
                     layer_input after sequence end
        """

        if layer_input.ndim != 3:
            raise ValueError("GRULayer.create_minibatch_structure() requires 3-dimensional input.")

        num_time_steps = layer_input.shape[0]
        num_sequences = layer_input.shape[1]
        self.layer_size = model_params['encoder_U_candidate'].shape[1]

        # Compute the gate pre-activations, which don't depend on the time step.
        x_preact_gates = \
                tensor.dot(layer_input, model_params['encoder_W_gates']) \
                + model_params['encoder_b_gates']
        x_preact_candidate = \
                tensor.dot(layer_input, model_params['encoder_W_candidate']) \
                + model_params['encoder_b_candidate']

        # The weights and biases for the previous step output. These have to be
        # applied inside the loop.
        U_gates = model_params['encoder_U_gates']
        U_candidate = model_params['encoder_U_candidate']

        sequences = [mask, x_preact_gates, x_preact_candidate]
        non_sequences = [U_gates, U_candidate]
        initial_hidden_state = tensor.unbroadcast(
            tensor.alloc(0.0, num_sequences, self.layer_size), 0)

        outputs, _ = theano.scan(
            self._create_time_step,
            sequences=sequences,
            outputs_info=[initial_hidden_state],
            non_sequences=non_sequences,
            name='hidden_layer_steps',
            n_steps=num_time_steps,
            profile=self._profile,
            strict=True)

        self.minibatch_output = outputs

    def create_onestep_structure(self, model_params, layer_input, state_input):
        """Creates GRU layer structure for one-step processing.

        This function is used for creating a text generator. The input is
        2-dimensional: the first dimension is the sequence and the second is
        the word projection.

        Sets self.onestep_outputs to a list of symbolic 2-dimensional matrices
        that describe the state outputs of the time steps. There's only one
        state in a GRU layer, h_(t).

        :type model_params: dict
        :param model_params: shared Theano variables

        :type layer_input: theano.tensor.var.TensorVariable
        :param layer_input: x_(t), symbolic 2-dimensional matrix that
                            describes the output of the previous layer (word
                            projections of the sequences)

        :type state_input: list of theano.tensor.var.TensorVariables
        :param state_input: a list of symbolic 2-dimensional matrices that
                            describe the state outputs of the previous time step
                            - only one state in a GRU layer, h_(t-1)
        """

        num_sequences = layer_input.shape[0]
        self.layer_size = model_params['encoder_U_candidate'].shape[1]

        mask = tensor.alloc(1.0, num_sequences, 1)

        # Compute the gate pre-activations, which don't depend on the time step.
        x_preact_gates = \
                tensor.dot(layer_input, model_params['encoder_W_gates']) \
                + model_params['encoder_b_gates']
        x_preact_candidate = \
                tensor.dot(layer_input, model_params['encoder_W_candidate']) \
                + model_params['encoder_b_candidate']

        hidden_state_input = state_input[0]

        # The weights and biases for the previous step output. These will
        # be applied inside _create_time_step().
        U_gates = model_params['encoder_U_gates']
        U_candidate = model_params['encoder_U_candidate']

        outputs = self._create_time_step(
            mask,
            x_preact_gates,
            x_preact_candidate,
            hidden_state_input,
            U_gates,
            U_candidate)
        self.onestep_outputs = [outputs]

    def _create_time_step(self, mask, x_preact_gates, x_preact_candidate, h_in, U_gates,
                          U_candidate):
        """The GRU step function for theano.scan(). Creates the structure of one
        time step.

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

        :type h_in: theano.tensor.var.TensorVariable
        :param h_in: h_(t-1), hidden state output of the previous time step

        :type U_gates: theano.tensor.var.TensorVariable
        :param U_gates: concatenation of the gate weights to be applied to
                        h_(t-1)

        :type U_candidate: theano.tensor.var.TensorVariable
        :param U_candidate: candidate state weight matrix to be applied to
                            h_(t-1)

        :rtype: theano.tensor.var.TensorVariable
        :returns: h_(t), the hidden state output
        """

        # pre-activation of the gates
        preact_gates = tensor.dot(h_in, U_gates)
        preact_gates += x_preact_gates

        # reset and update gates
        r = tensor.nnet.sigmoid(get_submatrix(preact_gates, 0, self.layer_size))
        u = tensor.nnet.sigmoid(get_submatrix(preact_gates, 1, self.layer_size))

        # pre-activation of the candidate state
        preact_candidate = tensor.dot(h_in, U_candidate)
        preact_candidate *= r
        preact_candidate += x_preact_candidate

        # hidden state output
        h_candidate = tensor.tanh(preact_candidate)
        h_out = (1.0 - u) * h_in + u * h_candidate

        # Apply the mask.
        h_out = mask[:, None] * h_out + (1.0 - mask)[:, None] * h_in

        return h_out
