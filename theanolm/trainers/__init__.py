import numpy
from theanolm.trainers.basictrainer import BasicTrainer
from theanolm.trainers.localstatisticstrainer import LocalStatisticsTrainer

def create_trainer(training_options, *args, **kwargs):
    """Constructs one of the BasicTrainer subclasses based on training options.

    :type training_options: dict
    :param training_options: a dictionary of training options
    """

    training_strategy = training_options['strategy']
    if training_strategy == 'basic':
        return BasicTrainer(training_options, *args, **kwargs)
    elif training_strategy == 'local-mean':
        return LocalStatisticsTrainer(
            training_options,
            *args,
            statistic_function=lambda x: numpy.mean(numpy.asarray(x)),
            **kwargs)
    elif training_strategy == 'local-median':
        return LocalStatisticsTrainer(
            training_options,
            *args,
            statistic_function=lambda x: numpy.median(numpy.asarray(x)),
            **kwargs)
    else:
        raise ValueError("Invalid training strategy requested: " + \
                         training_strategy)
