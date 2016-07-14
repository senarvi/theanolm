#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import theano

class RecurrentState:
    """State of recurrent layers at certain time step.

    When performing a forward pass one time step at a time, the state that has
    to be passed to the next time step includes the state of any recurrent
    layers.
    """

    def __init__(self, network, num_sequences=1):
        """Constructs a list of recurrent state variables and initializes them
        to zeros.

        This will create state vectors for each layer for each of the sequences
        that will be processed in parallel.

        :type network: Network
        :param network: the network object that contains the layers

        :type num_sequences: int
        :param num_sequences: number of sequences to be processed in parallel
        """

        self.sizes = network.recurrent_state_size
        self.num_sequences = num_sequences
        self.reset()

    def reset(self):
        """Resets the of recurrent state variables to zeros.

        The state vectors are 3-dimensional, because the layers support
        mini-batches with multiple sequences and time steps. Currently only one
        sequence can be computed at a time, when passing a state from time step
        to the next. Thus the first two dimensions have size 1.
        """

        self._state_arrays = []
        for size in self.sizes:
            shape = (1, self.num_sequences, size)
            value = numpy.zeros(shape).astype(theano.config.floatX)
            self._state_arrays.append(value)

    def set(self, state_arrays):
        """Sets the recurrent state variables.

        :type state_arrays: list of numpy.ndarrays
        :param state_arrays: state vector for each recurrent layer
        """

        if len(state_arrays) != len(self.sizes):
            raise ValueError("Recurrent state should contain as many arrays "
                             "as there are recurrent layers.")
        for x in state_arrays:
            if x.shape[0] != 1:
                raise ValueError("Recurrent state should contain only one time "
                                 "step.")
            if x.shape[1] != self.num_sequences:
                raise ValueError("Recurrent state contains incorrect number of "
                                 "sequences.")
        self._state_arrays = state_arrays

    def get(self):
        """Returns the recurrent state variables.
        """

        return self._state_arrays
