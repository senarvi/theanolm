#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.matrixfunctions import get_submatrix
from theanolm.layers.basiclayer import BasicLayer

class GRULayer(BasicLayer):
    """Gated Recurrent Unit Layer for Neural Network Language Model
    """

    def __init__(self, *args, **kwargs):
        """Initializes the parameters used by this layer.

        The weight matrices are concatenated so that they can be applied in a
        single parallel matrix operation. The same thing for bias vectors.
        """

        super().__init__(*args, is_recurrent=True, **kwargs)

        # The number of state variables to be passed between time steps.
        self.num_state_variables = 1

        # Initialize the parameters.
        input_size = self.input_layers[0].output_size
        output_size = self.output_size
        num_gates = 2
        # layer input weights for each gate and the candidate state
        self._init_orthogonal_weight('layer_input.W', input_size, output_size,
                                     scale=0.01, count=num_gates+1)
        # hidden state input weights for each gate and the candidate state
        self._init_orthogonal_weight('step_input.W', output_size, output_size,
                                     count=num_gates+1)
        # biases for each gate and the candidate state
        self._init_bias('layer_input.b', output_size, [None] * (num_gates + 1))

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
        that describe the state outputs. There's just one state in a GRU layer,
        h_(t). ``self.output`` will be set to the same hidden state output,
        which is also the actual output of this layer.

        Assumes that the shared variables have been passed using
        ``set_params()``.

        :type mask: theano.tensor.var.TensorVariable
        :param mask: symbolic 2-dimensional matrix that masks out time steps in
                     layer_input after sequence end

        :type state_inputs: list of theano.tensor.var.TensorVariables
        :param state_inputs: a list of symbolic 3-dimensional matrices that
                            describe the state outputs of the previous time step
                            - only one state in a GRU layer, h_(t-1)
        """

        input_matrix = self.input_layers[0].output
        num_time_steps = input_matrix.shape[0]
        num_sequences = input_matrix.shape[1]

        # Compute the gate and candidate state pre-activations, which don't
        # depend on the state input from the previous time step.
        layer_input_preact = self._tensor_preact(input_matrix, 'layer_input')

        # Weights of the hidden state input of each time step have to be applied
        # inside the loop.
        hidden_state_weights = self._get_param('step_input.W')

        if state_inputs is None:
            sequences = [mask, layer_input_preact]
            non_sequences = [hidden_state_weights]
            initial_value = numpy.dtype(theano.config.floatX).type(0.0)
            initial_hidden_state = \
                tensor.alloc(initial_value, num_sequences, self.output_size)

            hidden_state_output, _ = theano.scan(
                self._create_time_step,
                sequences=sequences,
                outputs_info=[initial_hidden_state],
                non_sequences=non_sequences,
                name='hidden_layer_steps',
                n_steps=num_time_steps,
                profile=self._profile,
                strict=True)
            self.state_outputs = [hidden_state_output]
        else:
            hidden_state_input = state_inputs[0]
    
            hidden_state_output = self._create_time_step(
                mask,
                layer_input_preact,
                hidden_state_input,
                hidden_state_weights)
            self.state_outputs = [hidden_state_output]

        self.output = self.state_outputs[0]

    def _create_time_step(self, mask, x_preact, h_in, h_weights):
        """The GRU step function for theano.scan(). Creates the structure of one
        time step.

        The inputs ``mask`` and ``x_preact`` contain only one time step, but
        possibly multiple sequences. There may, or may not be the first
        dimension of size 1 - it won't affect the computations, because
        broadcasting works by aligning the last dimensions.

        The required affine transformations have already been applied to the
        input prior to creating the loop. The transformed inputs and the mask
        that will be passed to the step function are vectors when processing a
        mini-batch - each value corresponds to the same time step in a different
        sequence.

        :type mask: theano.tensor.var.TensorVariable
        :param mask: a symbolic vector that masks out sequences that are past
                     the last word

        :type x_preact: theano.tensor.var.TensorVariable
        :param x_preact: concatenation of the input x_(t) pre-activations
                         computed using the gate and candidate state weights and
                         biases

        :type h_in: theano.tensor.var.TensorVariable
        :param h_in: h_(t-1), hidden state output of the previous time step

        :type h_weights: theano.tensor.var.TensorVariable
        :param h_weights: concatenation of the gate and candidate state weights
                          to be applied to h_(t-1)

        :rtype: theano.tensor.var.TensorVariable
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
