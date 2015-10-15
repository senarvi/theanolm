from theanolm.stoppers.basicstopper import BasicStopper
from theanolm.stoppers.significancestopper import SignificanceStopper
from theanolm.stoppers.learningratestopper import LearningRateStopper
from theanolm.stoppers.patiencestopper import PatienceStopper

def create_stopper(training_options, *args, **kwargs):
    """Constructs one of the BasicStopper subclasses based on training
    options.

    :type training_options: dict
    :param training_options: a dictionary of training options
    """

    stopping_criterion = training_options['stopping_criterion']
    if stopping_criterion == 'basic':
        return BasicStopper(training_options, *args, **kwargs)
    elif stopping_criterion == 'significance':
        return SignificanceStopper(training_options, *args, **kwargs)
    elif stopping_criterion == 'learning-rate':
        return LearningRateStopper(training_options, *args, **kwargs)
    elif stopping_criterion == 'patience':
        return PatienceStopper(training_options, *args, **kwargs)
    else:
        raise ValueError("Invalid stopping criterion requested: " + \
                         stopping_criterion)
