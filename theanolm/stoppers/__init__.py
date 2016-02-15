from theanolm.stoppers.basicstopper import BasicStopper
from theanolm.stoppers.noimprovementstopper import NoImprovementStopper
from theanolm.stoppers.annealingcountstopper import AnnealingCountStopper

def create_stopper(training_options, *args, **kwargs):
    """Constructs one of the BasicStopper subclasses based on training
    options.

    :type training_options: dict
    :param training_options: a dictionary of training options
    """

    stopping_criterion = training_options['stopping_criterion']
    if stopping_criterion == 'epoch-count':
        return BasicStopper(training_options, *args, **kwargs)
    elif stopping_criterion == 'no-improvement':
        return NoImprovementStopper(training_options, *args, **kwargs)
    elif stopping_criterion == 'annealing-count':
        return AnnealingCountStopper(training_options, *args, **kwargs)
    else:
        raise ValueError("Invalid stopping criterion requested: " + \
                         stopping_criterion)
