from theanolm.training.trainer import Trainer
from theanolm.training.sgdoptimizer import SGDOptimizer
from theanolm.training.nesterovoptimizer import NesterovOptimizer
from theanolm.training.adagradoptimizer import AdaGradOptimizer
from theanolm.training.adadeltaoptimizer import AdadeltaOptimizer
from theanolm.training.rmspropsgdoptimizer import RMSPropSGDOptimizer
from theanolm.training.rmspropnesterovoptimizer import RMSPropNesterovOptimizer
from theanolm.training.adamoptimizer import AdamOptimizer

def create_optimizer(optimization_options, *args, **kwargs):
    """Constructs one of the BasicOptimizer subclasses based on optimization
    options.

    :type optimization_options: dict
    :param optimization_options: a dictionary of optimization options

    :type network: Network
    :param network: the neural network object

    :type profile: bool
    :param profile: if set to True, creates a Theano profile object
    """

    optimization_method = optimization_options['method']
    if optimization_method == 'sgd':
        return SGDOptimizer(optimization_options, *args, **kwargs)
    elif optimization_method == 'nesterov':
        return NesterovOptimizer(optimization_options, *args, **kwargs)
    elif optimization_method == 'adagrad':
        return AdaGradOptimizer(optimization_options, *args, **kwargs)
    elif optimization_method == 'adadelta':
        return AdadeltaOptimizer(optimization_options, *args, **kwargs)
    elif optimization_method == 'rmsprop-sgd':
        return RMSPropSGDOptimizer(optimization_options, *args, **kwargs)
    elif optimization_method == 'rmsprop-nesterov':
        return RMSPropNesterovOptimizer(optimization_options, *args, **kwargs)
    elif optimization_method == 'adam':
        return AdamOptimizer(optimization_options, *args, **kwargs)
    else:
        raise ValueError("Invalid optimization method requested: " + \
                         optimization_method)
