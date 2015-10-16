#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import logging
import numpy
from theanolm.exceptions import IncompatibleStateError
from theanolm.stoppers.basicstopper import BasicStopper

class PatienceStopper(BasicStopper):
    """Criterion for stopping training when too many consequent validations have
    failed to show improvement.
    """

    def __init__(self, training_options, *args, **kwargs):
        """Creates a patience stopping criterion with given hyperparameters.
        
        :type training_options: dict
        :param training_options: dictionary of the training options, including
                                 options for early stopping
        """

        super().__init__(training_options, *args, **kwargs)

        self.min_epochs = training_options['min_epochs']
        self.stopping_patience = training_options['stopping_patience']
        self._wait_before_stopping = self.stopping_patience

    def start_new_minibatch(self):
        """Decides whether training should continue after the current
        mini-batch.

        :rtype: bool
        :returns: True if training should continue, False otherwise
        """

        if self.trainer.epoch_number <= self.min_epochs:
            return True

        if self.trainer.num_validations() == 0:
            return True

        if self.trainer.validations_since_min_cost() == 0:
            if self.trainer.has_improved():
                self._wait_before_stopping = self.stopping_patience
                logging.debug("Significant improvement. Will wait another %d "
                              "validations before stopping.",
                              self._wait_before_stopping)
        else:
            self._wait_before_stopping -= 1
        return self._wait_before_stopping > 0

    def get_state(self):
        """Returns the state of the stopping criterion.

        For consistency, all the parameter values are returned as numpy types,
        since state read from a model file also contains numpy types.

        :rtype: dict of numpy types
        :returns: a dictionary of the parameter values
        """

        result = OrderedDict()
        result['stopper.wait_before_stopping'] = \
            numpy.int64(self._wait_before_stopping)
        return result

    def set_state(self, state):
        """Sets the state of the stopping criterion.
        
        Requires that ``state`` contains values for all the state variables.

        :type state: dict of numpy types
        :param state: a dictionary of parameter values
        """

        if not 'stopper.wait_before_stopping' in state:
            raise IncompatibleStateError("Patience stopping condition state is "
                                         "missing.")
        self._wait_before_stopping = \
            state['stopper.wait_before_stopping'].item()
        logging.debug("stopper.wait_before_stopping <- " +
                      self._wait_before_stopping)
