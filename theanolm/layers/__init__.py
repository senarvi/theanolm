from theanolm.layers.networkinput import NetworkInput
from theanolm.layers.projectionlayer import ProjectionLayer
from theanolm.layers.tanhlayer import TanhLayer
from theanolm.layers.grulayer import GRULayer
from theanolm.layers.lstmlayer import LSTMLayer
from theanolm.layers.softmaxlayer import SoftmaxLayer

def create_layer(layer_type, *args, **kwargs):
    """Constructs one of the Layer classes based on a layer definition.

    :type layer_type: str
    :param layer_type: a text string describing the layer type
    """

    if layer_type == 'projection':
        return ProjectionLayer(*args, **kwargs)
    elif layer_type == 'tanh':
        return TanhLayer(*args, **kwargs)
    elif layer_type == 'lstm':
        return LSTMLayer(*args, **kwargs)
    elif layer_type == 'gru':
        return GRULayer(*args, **kwargs)
    elif layer_type == 'softmax':
        return SoftmaxLayer(*args, **kwargs)
    else:
        raise ValueError("Invalid layer type requested: " + layer_type)
