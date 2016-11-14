#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict

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

class BasicStopper(object):
    """Superclass for criteria to determine when to stop training. Deciding when
    to save the model is controlled by the trainer. The learning rate is
    controlled by the trainer and the optimization method, not by this class.

    This super class can be used directly, for a fixed number of epochs.
    """

    def __init__(self, training_options, trainer):
        """Creates a basic stopping criterion with given hyperparameters.

        :type training_options: dict
        :param training_options: dictionary of the training options, including
                                 options for early stopping

        :type trainer: theanolm.training.BasicTrainer or a subclass
        :param trainer: the trainer that will be used to get the status of the
                        training process
        """

        self.trainer = trainer
        self.max_epochs = training_options['max_epochs']

    def improvement_ceased(self):
        """Called when the performance of the model ceases to improve
        sufficiently on the validation set.
        """

        pass

    def start_new_epoch(self):
        """Decides whether training should continue with a new epoch.

        :rtype: bool
        :returns: True if a new epoch should be started, False otherwise
        """

        if self.trainer.epoch_number > self.max_epochs:
            print("Stopping because {} epochs was reached.".format(
                self.max_epochs))
            return False

        return self.start_new_minibatch()

    def start_new_minibatch(self):
        """Decides whether training should continue after the current
        mini-batch.

        :rtype: bool
        :returns: True if training should continue, False otherwise
        """

        return True

class NoImprovementStopper(BasicStopper):
    """Stops training when a better candidate state is not found between
    learning rate adjustments.

    Always waits that the entire training set has been passed at least
    min_epochs times, before stopping. Notice that the training might never
    stop, if min_epochs is never reached, when the performance won't improve and
    training is always returned to a point before that.
    """

    def __init__(self, training_options, *args, **kwargs):
        """Creates a significance stopping criterion with given hyperparameters.

        :type training_options: dict
        :param training_options: dictionary of the training options, including
                                 options for early stopping
        """

        super().__init__(training_options, *args, **kwargs)

        self.min_epochs = training_options['min_epochs']
        self._candidate_cost = None
        self._has_improved = True

    def improvement_ceased(self):
        """Called when the performance of the model ceases to improve
        sufficiently on the validation set. Checks if there was any improvement
        at all.
        """

        new_candidate_cost = self.trainer.candidate_cost()
        if new_candidate_cost is None:
            # No candidate state has been found.
            self._has_improved = False
            return

        if self._candidate_cost is None:
            # This is the first time the function is called.
            self._has_improved = True
            self._candidate_cost = new_candidate_cost
            return

        self._has_improved = new_candidate_cost < self._candidate_cost
        self._candidate_cost = new_candidate_cost

    def start_new_minibatch(self):
        """Decides whether training should continue after the current
        mini-batch.

        :rtype: bool
        :returns: True if training should continue, False otherwise
        """

        if self.trainer.epoch_number <= self.min_epochs:
            return True

        if self._has_improved:
            return True

        # Might be that improvement ceased earlier, but we have waited for the
        # minimum number of epochs to pass. During that time, we may have made
        # improvement.
        new_candidate_cost = self.trainer.candidate_cost()
        if not new_candidate_cost is None:
            if (self._candidate_cost is None) or \
               (new_candidate_cost < self._candidate_cost):
                self._has_improved = True
                return True

        return False

class AnnealingCountStopper(BasicStopper):
    """Stops training when the learning rate has been decreased
    a fixed number of times.
    """

    def __init__(self, training_options, *args, **kwargs):
        """Creates a learning rate stopping criterion with given
        hyperparameters.

        :type training_options: dict
        :param training_options: dictionary of the training options, including
                                 options for early stopping
        """

        super().__init__(training_options, *args, **kwargs)

        self.min_epochs = training_options['min_epochs']
        self._annealing_left = training_options['max_annealing_count']

    def improvement_ceased(self):
        """Called when the performance of the model ceases to improve
        sufficiently on the validation set.
        """

        self._annealing_left -= 1

    def start_new_minibatch(self):
        """Decides whether training should continue after the current
        mini-batch.

        :rtype: bool
        :returns: True if training should continue, False otherwise
        """

        if self.trainer.epoch_number <= self.min_epochs:
            return True

        return self._annealing_left >= 0
