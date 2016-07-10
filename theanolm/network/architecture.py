#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from theanolm.exceptions import IncompatibleStateError, InputError

class Architecture(object):
    """Neural Network Architecture Description
    
    A description of the neural network architecture can be read from a text
    file or from a neural network state stored in an HDF5 file.
    """

    def __init__(self, inputs, layers, output_layer=None):
        """Constructs a description of the neural network architecture.

        :type inputs: list of dict
        :param inputs: parameters for each input as a list of dictionaries

        :type layers: list of dict
        :param layers: parameters for each layer as a list of dictionaries

        :type output_layer: str
        :param output_layer: name of the layer that gives the output of the
                             network (the last layer if None)
        """

        self.inputs = inputs
        if not inputs:
            raise ValueError("Cannot construct Architecture without inputs.")

        self.layers = layers
        if not layers:
            raise ValueError("Cannot construct Architecture without layers.")

        if output_layer is None:
            self.output_layer = layers[-1]['name']
        else:
            self.output_layer = output_layer

    @classmethod
    def from_state(classname, state):
        """Constructs a description of the network architecture stored in a
        state.

        :type state: hdf5.File
        :param state: HDF5 file that contains the architecture parameters
        """

        if not 'architecture' in state:
            raise IncompatibleStateError(
                "Architecture is missing from neural network state.")
        h5_arch = state['architecture']

        if not 'inputs' in h5_arch:
            raise IncompatibleStateError(
                "Architecture parameter 'inputs' is missing from neural "
                "network state.")
        h5_inputs = h5_arch['inputs']
        inputs = []
        for input_id in sorted(h5_inputs.keys()):
            h5_input = h5_inputs[input_id]
            inputs.append(classname._read_h5_dict(h5_input))

        if not 'layers' in h5_arch:
            raise IncompatibleStateError(
                "Architecture parameter 'layers' is missing from neural "
                "network state.")
        h5_layers = h5_arch['layers']
        layers = []
        for layer_id in sorted(h5_layers.keys()):
            h5_layer = h5_layers[layer_id]
            layers.append(classname._read_h5_dict(h5_layer))

        if not 'output_layer' in h5_arch.attrs:
            raise IncompatibleStateError(
                "Architecture parameter 'output_layer' is missing from "
                "neural network state.")
        output_layer = h5_arch.attrs['output_layer']

        return classname(inputs, layers, output_layer)
    
    @classmethod
    def from_description(classname, description_file):
        """Reads a description of the network architecture from a text file.

        :type description_file: file or file-like object
        :param description_file: text file containing the description

        :rtype: Network.Architecture
        :returns: an object describing the network architecture
        """

        inputs = []
        layers = []

        for line in description_file:
            fields = line.split()
            if not fields:
                continue

            if fields[0] == 'input':
                input_description = dict()
                for field in fields[1:]:
                    variable, value = field.split('=')
                    input_description[variable] = value
                if not 'type' in input_description:
                    raise InputError("'type' is not given in an input description.")
                if not 'name' in input_description:
                    raise InputError("'name' is not given in an input description.")
                inputs.append(input_description)

            elif fields[0] == 'layer':
                layer_description = {'inputs': []}
                for field in fields[1:]:
                    variable, value = field.split('=')
                    if variable == 'size':
                        layer_description[variable] = int(value)
                    elif variable == 'input':
                        layer_description['inputs'].append(value)
                    else:
                        layer_description[variable] = value
                if not 'type' in layer_description:
                    raise InputError("'type' is not given in a layer description.")
                if not 'name' in layer_description:
                    raise InputError("'name' is not given in a layer description.")
                if not layer_description['inputs']:
                    raise InputError("'input' is not given in a layer description.")
                layers.append(layer_description)

            else:
                raise InputError("Invalid element in architecture "
                                 "description: {}".format(fields[0]))

        if not inputs:
            raise InputError("Architecture description contains no inputs.")
        if not layers:
            raise InputError("Architecture description contains no layers.")

        return classname(inputs, layers)

    @classmethod
    def from_package(classname, name):
        """Reads network architecture from one of the files packaged with
        TheanoLM.

        :type name: str
        :param name: name of a standard architecture file (without directory or
                     file extension)

        :rtype: Network.Architecture
        :returns: an object describing the network architecture
        """

        package_dir = os.path.abspath(os.path.dirname(__file__))
        description_path = os.path.join(package_dir,
                                        'architectures',
                                        name + '.arch')

        with open(description_path, 'rt', encoding='utf-8') as description_file:
            return classname.from_description(description_file)

    def get_state(self, state):
        """Saves the architecture parameters in a HDF5 file.

        The variable values will be saved as attributes of HDF5 groups. A
        group will be created for each level of the hierarchy.

        :type state: h5py.File
        :param state: HDF5 file for storing the architecture parameters
        """

        h5_arch = state.require_group('architecture')

        h5_inputs = h5_arch.require_group('inputs')
        for input_id, input in enumerate(self.inputs):
            h5_input = h5_inputs.require_group(str(input_id))
            self._write_h5_dict(h5_input, input)

        h5_layers = h5_arch.require_group('layers')
        for layer_id, layer in enumerate(self.layers):
            h5_layer = h5_layers.require_group(str(layer_id))
            self._write_h5_dict(h5_layer, layer)

        h5_arch.attrs['output_layer'] = self.output_layer

    def check_state(self, state):
        """Checks that the architecture stored in a state matches this
        network architecture, and raises an ``IncompatibleStateError``
        if not.

        :type state: h5py.File
        :param state: HDF5 file that contains the architecture parameters
        """

        if not 'architecture' in state:
            raise IncompatibleStateError(
                "Architecture is missing from neural network state.")
        h5_arch = state['architecture']

        if not 'inputs' in h5_arch:
            raise IncompatibleStateError(
                "Architecture parameter 'inputs' is missing from neural "
                "network state.")
        h5_inputs = h5_arch['inputs']
        for input_id, input in enumerate(self.inputs):
            h5_input = h5_inputs[str(input_id)]
            self._check_h5_dict(h5_input, input)

        if not 'layers' in h5_arch:
            raise IncompatibleStateError(
                "Architecture parameter 'layers' is missing from neural "
                "network state.")
        h5_layers = h5_arch['layers']
        for layer_id, layer in enumerate(self.layers):
            h5_layer = h5_layers[str(layer_id)]
            self._check_h5_dict(h5_layer, layer)

        if not 'output_layer' in h5_arch.attrs:
            raise IncompatibleStateError(
                "Architecture parameter 'output_layer' is missing from "
                "neural network state.")
        h5_value = h5_arch.attrs['output_layer']
        if self.output_layer != h5_value:
            raise IncompatibleStateError(
                "Neural network state has output_layer={1}, while "
                "this architecture has output_layer={0}.".format(
                    self.output_layer, h5_value))

    @staticmethod
    def _read_h5_dict(h5_dict):
        """Reads a dictionary from a HDF5 file.

        The dictionary can store strings, which are represented as HDF5
        attributes, and lists of strings, which are represented as attributes of
        HDF5 subgroups.

        :type h5_dict: h5py.Group
        :param h5_dict: The HD5 group that stores a dictionary of strings in its
                        attributes
        """

        result = dict()
        for variable in h5_dict.attrs:
            result[variable] = h5_dict.attrs[variable]
        for variable in h5_dict:
            values = []
            h5_values = h5_dict[variable]
            for value_id in sorted(h5_values.attrs.keys()):
                values.append(h5_values.attrs[value_id])
            result[variable] = values
        return result

    @staticmethod
    def _write_h5_dict(h5_group, variables):
        """Writes a dictionary to a HDF5 file.

        The dictionary can store strings, which are represented as HDF5
        attributes, and lists of strings, which are represented as attributes of
        HDF5 subgroups.

        :type h5_group: h5py.Group
        :param h5_group: The HD5 group that will store a dictionary of strings
                         in its attributes

        :type variables: dict
        :param variables: a dictionary that may contain strings and lists of strings
        """

        for variable, values in variables.items():
            if isinstance(values, list):
                h5_values = h5_group.require_group(variable)
                for value_id, value in enumerate(values):
                    h5_values.attrs[str(value_id)] = value
            else:
                h5_group.attrs[variable] = values

    @staticmethod
    def _check_h5_dict(h5_group, variables):
        """Checks that a dictionary matches a HDF5 group and raises an
        ``IncompatibleStateError`` if not.

        The dictionary can store strings, which are represented as HDF5
        attributes, and lists of strings, which are represented as attributes of
        HDF5 subgroups.

        :type h5_group: h5py.Group
        :param h5_group: The HD5 group that will store a dictionary of strings
                         in its attributes

        :type variables: dict
        :param variables: a dictionary that may contain strings and lists of strings
        """

        for variable, values in variables.items():
            if isinstance(values, list):
                h5_values = h5_group[variable]
                for value_id, value in enumerate(values):
                    h5_value = h5_values.attrs[str(value_id)]
                    if value != h5_value:
                        raise IncompatibleStateError(
                            "Neural network state has {0}={2}, while this "
                            "architecture has {0}={1}.".format(
                                variable, value, h5_value))
            else:
                h5_value = h5_group.attrs[variable]
                if values != h5_value:
                    raise IncompatibleStateError(
                        "Neural network state has {0}={2}, while this "
                        "architecture has {0}={1}.".format(
                            variable, value, h5_value))
