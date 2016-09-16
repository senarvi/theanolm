#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import theano

class RecurrentState:
    """State of Recurrent Layers at Certain Time Step

    Recurrent layers contain one or more state variables (vectors). When
    performing a forward pass one time step at a time, these state variables
    have to be saved and passed as input to the next time step. A
    ``RecurrentState`` object contains a matrix for each recurrent
    state variable.

    Every matrix has the same shape as a mini-batch: First dimension is the time
    step. Since one time step is processed at a time, the size of the first
    dimension is one. The second dimension is sequence. Multiple sequences may
    be processed in parallel. The third dimension is the state vector.
    """

    def __init__(self, sizes, num_sequences=1, state_variables=None):
        """Constructs a list of recurrent layer states and initializes them.

        This will create state vectors for each layer for each of the sequences
        that will be processed in parallel. Unless ``state_variables`` is given,
        the vectors will be initialized to zeros.

        :type sizes: list of ints
        :param sizes: size of each recurrent layer state

        :type num_sequences: int
        :param num_sequences: number of sequences to be processed in parallel

        :type state_variables: list of numpy.ndarrays
        :param state_variables: if set to other than ``None``, sets the initial
                                recurrent layer states to this instead of zeros
        """

        self.sizes = sizes
        self.num_sequences = num_sequences
        if state_variables is None:
            self.reset()
        else:
            self.set(state_variables)

    @classmethod
    def combine_sequences(classname, states):
        """Creates recurrent state variables that combine all the sequences from
        the given list of state objects.

        Takes multiple ``RecurrentState`` objects and creates one that contains
        a state matrix or matrices that combine all the sequences. The resulting
        state will contain as many sequences as the input states in total, in
        the same order as in the ``states`` list. The purpose is to be able to
        process multiple forward passes in parallel.

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
        state_variables = []
        num_variables = len(states[0].sizes)
        for index in range(num_variables):
            state_variable = [state.get(index) for state in states]
            state_variable = numpy.concatenate(state_variable, axis=1)
            assert state_variable.shape[0] == 1
            assert state_variable.shape[1] == num_sequences
            assert state_variable.shape[2] == states[0].sizes[index]
            sizes.append(state_variable.shape[2])
            state_variables.append(state_variable)

        return classname(sizes, num_sequences, state_variables)

    def reset(self):
        """Resets the state of each recurrent layer to zeros.

        The state vectors are 3-dimensional, because the layers support
        mini-batches with multiple sequences and time steps. Currently only one
        sequence can be computed at a time, when passing a state from time step
        to the next. Thus the first two dimensions have size 1.
        """

        self._state_variables = []
        for size in self.sizes:
            shape = (1, self.num_sequences, size)
            value = numpy.zeros(shape).astype(theano.config.floatX)
            self._state_variables.append(value)

    def set(self, state_variables):
        """Sets the state vector of every recurrent layer.

        :type state_variables: list of numpy.ndarrays
        :param state_variables: a matrix for each recurrent layer that contains
                                the state vector for each sequence at one time
                                step
        """

        if len(state_variables) != len(self.sizes):
            raise ValueError("Recurrent state should contain as many arrays "
                             "as there are recurrent layers.")
        for state_variable, size in zip(state_variables, self.sizes):
            if state_variable.shape[0] != 1:
                raise ValueError("Recurrent state should contain only one time "
                                 "step.")
            if state_variable.shape[1] != self.num_sequences:
                raise ValueError("Recurrent state contains incorrect number of "
                                 "sequences.")
            if state_variable.shape[2] != size:
                raise ValueError("Recurrent state contains a layer with "
                                 "incorrect size.")
        self._state_variables = state_variables

    def get(self, index=None):
        """Returns the state matrix of a given layer, or a list of the matrices
        of all layers.

        :type index: int
        :param index: index of a recurrent layer in the state object; if
                            set to other than ``None``, returns only the matrix
                            for the corresponding layer

        :rtype: numpy.ndarray or list of numpy.ndarrays
        :returns: a matrix for each recurrent layer that contains the state
                  vector for each sequence at one time step
        """

        if index is not None:
            return self._state_variables[index]
        else:
            return self._state_variables
