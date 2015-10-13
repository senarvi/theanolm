from theanolm.trainers.basictrainer import BasicTrainer
from theanolm.trainers.medianvalidationtrainer import MedianValidationTrainer
from theanolm.trainers.meanvalidationtrainer import MeanValidationTrainer
from theanolm.trainers.validationaveragetrainer import ValidationAverageTrainer

def create_trainer(training_options, *args, **kwargs):
    """Constructs one of the BasicTrainer subclasses based on training options.

    :type training_options: dict
    :param training_options: a dictionary of training options
    """

    training_strategy = training_options['strategy']
    if training_strategy == 'basic':
        return BasicTrainer(training_options, *args, **kwargs)
    elif training_strategy == 'local-mean':
        return MeanValidationTrainer(training_options, *args, **kwargs)
    elif training_strategy == 'local-median':
        return MedianValidationTrainer(training_options, *args, **kwargs)
    elif training_strategy == 'validation-average':
        return ValidationAverageTrainer(training_options, *args, **kwargs)
    else:
        raise ValueError("Invalid training strategy requested: " + \
                         training_strategy)
