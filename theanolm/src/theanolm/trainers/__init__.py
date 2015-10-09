from theanolm.trainers.sgdtrainer import SGDTrainer
from theanolm.trainers.nesterovtrainer import NesterovTrainer
from theanolm.trainers.adadeltatrainer import AdadeltaTrainer
from theanolm.trainers.rmspropsgdtrainer import RMSPropSGDTrainer
from theanolm.trainers.rmspropmomentumtrainer import RMSPropMomentumTrainer
from theanolm.trainers.adamtrainer import AdamTrainer

def create_trainer(optimization_method, network, training_options,
                   profile=False):
    """Constructs one of the ModelTrainer subclasses based on a string argument.

    :type optimization_method: str
    :param optimization_method: a string identifying the correct subclass

    :type network: Network
    :param network: the neural network object

    :type training_options: dict
    :param training_options: a dictionary of training options

    :type profile: bool
    :param profile: if set to True, creates a Theano profile object
    """

    if optimization_method == 'sgd':
        return SGDTrainer(network, training_options, profile)
    elif optimization_method == 'nesterov':
        return NesterovTrainer(network, training_options, profile)
    elif optimization_method == 'adadelta':
        return AdadeltaTrainer(network, training_options, profile)
    elif optimization_method == 'rmsprop-sgd':
        return RMSPropSGDTrainer(network, training_options, profile)
    elif optimization_method == 'rmsprop-momentum':
        return RMSPropMomentumTrainer(network, training_options, profile)
    elif optimization_method == 'adam':
        return AdamTrainer(network, training_options, profile)
    else:
        raise ValueError("Invalid optimization method requested: " + \
                         optimization_method)
