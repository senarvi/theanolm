#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import theano
from theanolm.exceptions import IncompatibleStateError, TheanoConfigurationError

class Parameters:
    """Theano Function Parameters

    A dictionary of Theano shared variables. The values can be accessed through
    their path, which also acts as their identifier when they are saved to a
    HDF5 file.
    """

    def __init__(self):
        """Initializes an empty parameter dictionary.
        """

        self._vars = dict()
        self.total_size = 0

    def __getitem__(self, path):
        """Returns a shared variable given parameter path.

        :type path: str
        :param path: parameter path

        :rtype: SharedVariable
        :returns: the corresponding Theano shared variable
        """

        return self._vars[path]

    def add(self, path, value, device=None):
        """Adds a new parameter.

        :type path: str
        :param path: identifier for the shared variable in Theano and its value
                     when stored in a HDF5 file

        :type value: numpy.ndarray
        :param value: initial value for the shared variable

        :type device: str
        :param device: if other than ``None``, the shared variable will be
                       kept in this device
        """

        if path in self._vars:
            raise ValueError("Path `{}' already in parameters.".format(path))
        if theano.config.device.startswith('gpu') and value.dtype == 'float64':
            raise TheanoConfigurationError(
                'You are using Theano with the old GPU backend ("device=gpu"), '
                'and the parameter {} is float64. This is very inefficient, so '
                'you most likely want to set "floatX=float32".'.format(path))

        if device is None:
            self._vars[path] = theano.shared(value, path)
        else:
            try:
                self._vars[path] = theano.shared(value, path, target=device)
            except TypeError:
                raise RuntimeError(
                    "Unable to create Theano shared variable for parameter {} "
                    "on device {}. If you are using the old backend, you "
                    "cannot assign layers to different GPU devices."
                    .format(path, device))

        logging.debug("     * %s size=%d type=%s device=%s",
                      path, value.size, value.dtype, str(device))
        self.total_size += value.size

    def get_state(self, state):
        """Pulls values from the shared variables into a HDF5 file.

        If there already is a parameter in the file, it will be replaced, so it
        has to have the same number of elements.

        :type state: h5py.File
        :param state: HDF5 file for storing the parameters
        """

        for path, param in self._vars.items():
            if path in state:
                state[path][:] = param.get_value()
            else:
                state.create_dataset(path, data=param.get_value())

    def set_state(self, state):
        """Sets the values of the shared variables.

        Requires that ``state`` contains values for all the parameters.

        :type state: h5py.File
        :param state: HDF5 file that contains the parameters
        """

        for path, param in self._vars.items():
            if not path in state:
                raise IncompatibleStateError(
                    "Parameter `%s' is missing from state." % path)
            new_value = state[path].value
            param.set_value(new_value)
            if len(new_value.shape) == 0:
                logging.debug("%s <- %s", path, str(new_value))
            else:
                logging.debug("%s <- array%s", path, str(new_value.shape))

    def get_variables(self):
        """Returns a list of the shared variables.

        :rtype: list of strs
        :returns: parameter paths
        """

        return self._vars
