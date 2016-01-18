from theanolm.layers.networkinput import NetworkInput
from theanolm.layers.projectionlayer import ProjectionLayer
from theanolm.layers.tanhlayer import TanhLayer
from theanolm.layers.grulayer import GRULayer
from theanolm.layers.lstmlayer import LSTMLayer
from theanolm.layers.softmaxlayer import SoftmaxLayer
from theanolm.layers.dropoutlayer import DropoutLayer

def create_layer(layer_options, *args, **kwargs):
    """Constructs one of the Layer classes based on a layer definition.

    :type layer_type: str
    :param layer_type: a text string describing the layer type
    """

    layer_type = layer_options['type']
    if layer_type == 'projection':
        return ProjectionLayer(layer_options, *args, **kwargs)
    elif layer_type == 'tanh':
        return TanhLayer(layer_options, *args, **kwargs)
    elif layer_type == 'lstm':
        return LSTMLayer(layer_options, *args, **kwargs)
    elif layer_type == 'gru':
        return GRULayer(layer_options, *args, **kwargs)
    elif layer_type == 'softmax':
        return SoftmaxLayer(layer_options, *args, **kwargs)
    elif layer_type == 'dropout':
        return DropoutLayer(layer_options, *args, **kwargs)
    else:
        raise ValueError("Invalid layer type requested: " + layer_type)
