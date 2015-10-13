#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class BasicStoppingCriterion(object):
    """Superclass for criteria to determine when to stop training. Deciding when
    to save the model is controlled by the caller. The learning rate is
    controlled by the caller and the optimization method, not by this class.

    This super class can be used directly, for a fixed number of epochs.
    """

    def __init__(self, training_options, trainer):
        """
        :type trainer: theanolm.training.Trainer
        """

        self.trainer = trainer
        self.max_epochs = training_options['max_epochs']

    def start_new_epoch(self):
        """Should training continue with a new epoch?

        :rtype: bool
        :returns: True if a new epoch should be started
        """

        if self.trainer.epoch_number > self.max_epochs:
            print("Stopping because {} epochs was reached.".format(
                self.max_epochs))
            return False

        return self.start_new_minibatch()

    def start_new_minibatch(self):
        """Should training continue after the current minibatch?
        """

        return True

    def learning_rate_decreased(self):
        pass

class LearningRateStoppingCriterion(BasicStoppingCriterion):
    """Stops training when the learning rate has been decreased
    a fixed number of times.
    """

    def __init__(self, training_options, *args, **kwargs):
        super().__init__(training_options, *args, **kwargs)

        self.min_epochs = training_options['min_epochs']
        self.decreases_left = training_options['max_decreases']

    def learning_rate_decreased(self):
        self.decreases_left -= 1

    def start_new_minibatch(self, trainer):
        if self.trainer.epoch_number <= self.min_epochs:
            return True

        return self.decreases_left > 0

class PatienceStoppingCriterion(BasicStoppingCriterion):
    """Stops training when too many consequent validations have
    failed to show improvement.
    """

    # FIXME: relation between stopping patience and learning rate patience?
    def __init__(self, training_options, *args, **kwargs):
        super().__init__(training_options, *args, **kwargs)

        self.min_epochs = training_options['min_epochs']
        self.max_patienece = training_options['patience']
        self.current_patience = self.max_patience

    def start_new_minibatch(self, trainer):
        if self.trainer.epoch_number <= self.min_epochs:
            return True

        if self.trainer.validations_since_min_cost() == 0:
            self.current_patience = self.max_patienece
        return self.current_patience > 0

def create_stopper(training_options, *args, **kwargs):
    """Constructs one of the BasicStoppingCriterion subclasses based on training
    options.

    :type training_options: dict
    :param training_options: a dictionary of training options
    """

    stopping_criterion = training_options['stopping_criterion']
    if stopping_criterion == 'basic':
        return BasicStoppingCriterion(training_options, *args, **kwargs)
    elif stopping_criterion == 'learning-rate':
        return LearningRateStoppingCriterion(training_options, *args, **kwargs)
    elif stopping_criterion == 'patience':
        return PatienceStoppingCriterion(training_options, *args, **kwargs)
    else:
        raise ValueError("Invalid stopping criterion requested: " + \
                         stopping_criterion)
