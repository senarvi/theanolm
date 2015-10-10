from theanolm.optimizers.sgdoptimizer import SGDOptimizer
from theanolm.optimizers.nesterovoptimizer import NesterovOptimizer
from theanolm.optimizers.adadeltaoptimizer import AdadeltaOptimizer
from theanolm.optimizers.rmspropsgdoptimizer import RMSPropSGDOptimizer
from theanolm.optimizers.rmspropmomentumoptimizer import RMSPropMomentumOptimizer
from theanolm.optimizers.adamoptimizer import AdamOptimizer

def create_optimizer(network, optimization_options,
                     profile=False):
    """Constructs one of the BasicOptimizer subclasses based on a string argument.

    :type network: Network
    :param network: the neural network object

    :type optimization_options: dict
    :param optimization_options: a dictionary of optimization options

    :type profile: bool
    :param profile: if set to True, creates a Theano profile object
    """

    optimization_method = optimization_options['method']
    if optimization_method == 'sgd':
        return SGDOptimizer(network, optimization_options, profile)
    elif optimization_method == 'nesterov':
        return NesterovOptimizer(network, optimization_options, profile)
    elif optimization_method == 'adadelta':
        return AdadeltaOptimizer(network, optimization_options, profile)
    elif optimization_method == 'rmsprop-sgd':
        return RMSPropSGDOptimizer(network, optimization_options, profile)
    elif optimization_method == 'rmsprop-momentum':
        return RMSPropMomentumOptimizer(network, optimization_options, profile)
    elif optimization_method == 'adam':
        return AdamOptimizer(network, optimization_options, profile)
    else:
        raise ValueError("Invalid optimization method requested: " + \
                         optimization_method)
