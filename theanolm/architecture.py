#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from theanolm.exceptions import IncompatibleStateError, InputError

class Architecture(object):
    """Neural Network Architecture Description
    
    A description of the neural network architecture can be read from a text
    file or from a neural network state stored in an HDF5 file.
    """

    def __init__(self, layers, output_layer=None):
        """Constructs a description of the neural network architecture.

        :type layers: list of dict
        :param layers: parameters for each layer as a list of dictionaries
        
        :type output_layer: str
        :param output_layer: name of the layer that gives the output of the
                             network (the last layer if None)
        """

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

        layers = []

        if not 'arch' in state:
            raise IncompatibleStateError(
                "Architecture is missing from neural network state.")
        h5_arch = state['arch']

        if not 'output_layer' in h5_arch.attrs:
            raise IncompatibleStateError(
                "Architecture parameter 'output_layer' is missing from "
                "neural network state.")
        output_layer = h5_arch.attrs['output_layer']

        if not 'layers' in h5_arch:
            raise IncompatibleStateError(
                "Architecture parameter 'layers' is missing from neural "
                "network state.")
        h5_layers = h5_arch['layers']

        for layer_id in sorted(h5_layers.keys()):
            layer = dict()
            h5_layer = h5_layers[layer_id]
            for variable in h5_layer.attrs:
                layer[variable] = h5_layer.attrs[variable]
            for variable in h5_layer:
                values = []
                h5_values = h5_layer[variable]
                for value_id in sorted(h5_values.attrs.keys()):
                    values.append(h5_values.attrs[value_id])
                layer[variable] = values
            layers.append(layer)

        return classname(layers, output_layer)
    
    @classmethod
    def from_description(classname, description_file):
        """Reads a description of the network architecture from a text file.

        :type description_file: file or file-like object
        :param description_file: text file containing the description

        :rtype: Network.Architecture
        :returns: an object describing the network architecture
        """

        layers = []

        for line in description_file:
            fields = line.split()
            if not fields:
                continue
            if fields[0] != 'layer':
                raise InputError("Expecting 'layer' in architecture "
                                 "description.")

            layer_description = { 'inputs': [] }
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

        if not layers:
            raise InputError("Architecture description is empty.")

        return classname(layers)

    def get_state(self, state):
        """Saves the architecture parameters in a HDF5 file.

        The variable values will be saved as attributes of HDF5 groups. A
        group will be created for each level of the hierarchy.

        :type state: h5py.File
        :param state: HDF5 file for storing the architecture parameters
        """

        h5_arch = state.require_group('arch')
        h5_arch.attrs['output_layer'] = self.output_layer

        h5_layers = h5_arch.require_group('layers')
        for layer_id, layer in enumerate(self.layers):
            h5_layer = h5_layers.require_group(str(layer_id))
            for variable, values in layer.items():
                if isinstance(values, list):
                    h5_values = h5_layer.require_group(variable)
                    for value_id, value in enumerate(values):
                        h5_values.attrs[str(value_id)] = value
                else:
                    h5_layer.attrs[variable] = values

    def check_state(self, state):
        """Checks that the architecture stored in a state matches this
        network architecture, and raises an ``IncompatibleStateError``
        if not.

        :type state: h5py.File
        :param state: HDF5 file that contains the architecture parameters
        """

        if not 'arch' in state:
            raise IncompatibleStateError(
                "Architecture is missing from neural network state.")
        h5_arch = state['arch']

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

        if not 'layers' in h5_arch:
            raise IncompatibleStateError(
                "Architecture parameter 'layers' is missing from neural "
                "network state.")
        h5_layers = h5_arch['layers']
        for layer_id, layer in enumerate(self.layers):
            h5_layer = h5_layers[str(layer_id)]
            for variable, values in layer.items():
                if isinstance(values, list):
                    h5_values = h5_layer[variable]
                    for value_id, value in enumerate(values):
                        h5_value = h5_values.attrs[str(value_id)]
                        if value != h5_value:
                            raise IncompatibleStateError(
                                "Neural network state has {0}={2}, while "
                                "this architecture has {0}={1}.".format(
                                    variable, value, h5_value))
                else:
                    h5_value = h5_layer.attrs[variable]
                    if values != h5_value:
                        raise IncompatibleStateError(
                            "Neural network state has {0}={2}, while "
                            "this architecture has {0}={1}.".format(
                                variable, value, h5_value))
