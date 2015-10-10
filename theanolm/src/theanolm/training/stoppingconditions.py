#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class BasicStoppingCondition(object):
    """Superclass for conditions to determine when to stop training.
    Deciding when to save the model is controlled by the caller.
    The learning rate is controlled by the caller and trainer,
    not by this class.

    This class can be used directly, for a fixed number of epochs.
    """

    def __init__(self, trainer, training_options):
        """
        :type trainer: theanolm.training.Trainer
        """

        self.process = trainer
        self.max_epochs = training_options['max_epochs']

    def start_new_epoch(self):
        """Should training continue with a new epoch?

        :rtype: bool
        :returns: True if a new epoch should be started
        """

        if self.process.epoch_number > self.max_epochs:
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

class LearningRateStoppingCondition(BasicStoppingCondition):
    """Stops training when the learning rate has been decreased
    a fixed number of times.
    """

    def __init__(self, trainer, training_options):
        super().__init__(trainer, training_options)

        self.min_epochs = training_options['min_epochs']
        self.decreases_left = training_options['max_decreases']

    def learning_rate_decreased(self):
        self.decreases_left -= 1

    def start_new_minibatch(self, trainer):
        if self.process.epoch_number <= self.min_epochs:
            return True

        return self.decreases_left > 0

class PatienceStoppingCondition(BasicStoppingCondition):
    """Stops training when too many consequent validations have
    failed to show improvement.
    """

    # FIXME: relation between stopping patience and learning rate patience?
    def __init__(self, trainer, training_options):
        super().__init__(trainer, training_options)

        self.min_epochs = training_options['min_epochs']
        self.max_patienece = training_options['patience']
        self.current_patience = self.max_patience

    def start_new_minibatch(self, trainer):
        if self.process.epoch_number <= self.min_epochs:
            return True

        if self.process.validations_since_min_cost() == 0:
            self.current_patience = self.max_patienece
        return self.current_patience > 0
