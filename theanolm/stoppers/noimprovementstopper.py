#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from theanolm.stoppers.basicstopper import BasicStopper

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
