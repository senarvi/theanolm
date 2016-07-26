#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import theano

class RecurrentState:
    """State of Recurrent Layers at Certain Time Step

    When performing a forward pass one time step at a time, the state that has
    to be passed to the next time step includes the state of any recurrent
    layers.
    """

    def __init__(self, sizes, num_sequences=1, layer_states=None):
        """Constructs a list of recurrent layer states and initializes them.

        This will create state vectors for each layer for each of the sequences
        that will be processed in parallel. Unless ``layer_states`` is given,
        the vectors will be initialized to zeros.

        :type sizes: list of ints
        :param sizes: size of each recurrent layer state

        :type num_sequences: int
        :param num_sequences: number of sequences to be processed in parallel

        :type layer_states: list of numpy.ndarrays
        :param layer_states: if set to other than ``None``, sets the initial
                             recurrent layer states to this instead of zeros
        """

        self.sizes = sizes
        self.num_sequences = num_sequences
        if layer_states is None:
            self.reset()
        else:
            self.set(layer_states)

    @classmethod
    def combine_sequences(classname, states):
        """Creates recurrent layer state that combines all the sequences from
        the given state matrices.

        Takes multiple ``RecurrentState`` objects and creates one that contains
        a matrix or matrices that combine all the sequences. The resulting state
        will contain as many sequences as the input states in total, in the same
        order as in the ``states`` list.

        :type states: list of RecurrentStates
        :param states: list of recurrent layer states, each containing N1, N2,
                       N3, ... sequences

        :rtype: RecurrentState
        :returns: a recurrent layer state that contains the N1 + N2 + N3 + ...
                  sequences from ``states``, in the same order that they appear
                  in ``states``
        """

        if not states:
            raise ValueError("Need at least one RecurrentState object.")

        num_sequences = sum([state.num_sequences for state in states])

        sizes = []
        layer_states = []
        num_layers = len(states[0].sizes)
        for layer_index in range(num_layers):
            layer_state = [state._layers[layer_index] for state in states]
            layer_state = numpy.concatenate(layer_state, axis=1)
            assert layer_state.shape[0] == 1
            assert layer_state.shape[1] == num_sequences
            assert layer_state.shape[2] == states[0].sizes[layer_index]
            sizes.append(layer_state.shape[2])
            layer_states.append(layer_state)

        return classname(sizes, num_sequences, layer_states)

    def reset(self):
        """Resets the state of each recurrent layer to zeros.

        The state vectors are 3-dimensional, because the layers support
        mini-batches with multiple sequences and time steps. Currently only one
        sequence can be computed at a time, when passing a state from time step
        to the next. Thus the first two dimensions have size 1.
        """

        self._layers = []
        for size in self.sizes:
            shape = (1, self.num_sequences, size)
            value = numpy.zeros(shape).astype(theano.config.floatX)
            self._layers.append(value)

    def set(self, layer_states):
        """Sets the state vector of every recurrent layer.

        :type layer_states: list of numpy.ndarrays
        :param layer_states: a matrix for each recurrent layer that contains the
                             state vector for each sequence at one time step
        """

        if len(layer_states) != len(self.sizes):
            raise ValueError("Recurrent state should contain as many arrays "
                             "as there are recurrent layers.")
        for x in layer_states:
            if x.shape[0] != 1:
                raise ValueError("Recurrent state should contain only one time "
                                 "step.")
            if x.shape[1] != self.num_sequences:
                raise ValueError("Recurrent state contains incorrect number of "
                                 "sequences.")
        self._layers = layer_states

    def get(self):
        """Returns a list of state vectors, one for every recurrent layer.

        :rtype: list of numpy.ndarrays
        :returns: a matrix for each recurrent layer that contains the state
                  vector for each sequence at one time step
        """

        return self._layers
