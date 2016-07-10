#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import theano

class RecurrentState:
    """State of recurrent layers at certain time step.

    When performing a forward pass one time step at a time, the state that has
    to be passed to the next time step includes the state of any recurrent
    layers.

    Currently multiple sequences cannot be computed in parallel.
    """

    def __init__(self, network):
        self.sizes = network.recurrent_state_size
        self.reset()

    def reset(self):
        """Constructs a list of recurrent state variables and initializes them
        to zeros.

        The state vectors are 3-dimensional, because the layers support
        mini-batches with multiple sequences and time steps. Currently only one
        sequence can be computed at a time, when passing a state from time step
        to the next. Thus the first two dimensions have size 1.
        """

        self._state = []
        for size in self.sizes:
            shape = (1, 1, size)
            value = numpy.zeros(shape).astype(theano.config.floatX)
            self._state.append(value)

    def set(self, x):
        """Sets the recurrent state variables.

        :type x: list of numpy.ndarrays
        :param x: state vector for each recurrent layer
        """

        if len(x) != len(self.sizes):
            raise ValueError("Recurrent state should contain as many vectors "
                             "as there are recurrent layers.")
        self._state = x

    def get(self):
        """Returns the recurrent state variables.
        """

        return _state
