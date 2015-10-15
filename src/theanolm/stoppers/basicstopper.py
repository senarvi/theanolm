#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict

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
    
    def get_state(self):
        """Returns the state of the stopping criterion.

        For consistency, all the parameter values are returned as numpy types,
        since state read from a model file also contains numpy types.

        :rtype: dict of numpy types
        :returns: a dictionary of the parameter values
        """

        result = OrderedDict()
        return result

    def set_state(self, state):
        """Sets the state of the stopping criterion.
        
        Requires that ``state`` contains values for all the state variables.

        :type state: dict of numpy types
        :param state: a dictionary of parameter values
        """

        pass
