#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from theanolm.stoppers.basicstopper import BasicStopper

class SignificanceStopper(BasicStopper):
    """Stops training when the validation cost is not improved enough between
    learning rate reductions.
    """

    def __init__(self, training_options, *args, **kwargs):
        """Creates a significance stopping criterion with given hyperparameters.

        :type training_options: dict
        :param training_options: dictionary of the training options, including
                                 options for early stopping
        """

        super().__init__(training_options, *args, **kwargs)

        self.min_epochs = training_options['min_epochs']
        self._has_improved = True

    def improvement_ceased(self):
        """Called when the performance of the model ceases to improve
        sufficiently on the validation set.
        """

        # The function is not called before any validations have been performed.
        assert self.trainer.num_validations() > 0
        
        self._has_improved = self.trainer.has_improved()

    def start_new_minibatch(self):
        """Decides whether training should continue after the current
        mini-batch.

        :rtype: bool
        :returns: True if training should continue, False otherwise
        """

        if self.trainer.epoch_number <= self.min_epochs:
            return True

        return self._has_improved
