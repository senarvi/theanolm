#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from theanolm.stoppers.basicstopper import BasicStopper

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
