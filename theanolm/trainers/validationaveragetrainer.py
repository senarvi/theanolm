#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy
from theanolm.trainers.basictrainer import BasicTrainer

class ValidationAverageTrainer(BasicTrainer):
    """A trainer that computes the average of the perplexity from three
    consecutive validations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous_state = None

    def _validate(self, perplexity):
        """When ``perplexity`` is not None, appends it to cost history and
        validates whether there was improvement.

        :type perplexity: float
        :param perplexity: computed perplexity at a validation point, None
                           elsewhere
        """

        if perplexity is None:
            return

        self._append_validation_cost(perplexity)

        validations_since_best = self.validations_since_min_cost()
        if validations_since_best == 0:
            # At least three validations have been performed and this is the
            # minimum cost so far.
            assert not self.previous_state is None

            # The minimum cost is from the three previous validations.
            # Previous validation is in the middle of those validations.
            self._set_min_cost_state(self.previous_state)
        else:
            self.previous_state = self.get_state()
            if (self.options['annealing_patience'] >= 0) and \
               (validations_since_best > self.options['annealing_patience']):
                # Too many validations without improvement.

                # If any validations have been done, the best state has been found
                # and saved. If training has been started from previous state,
                # min_cost_state has been set to the initial state.
                assert not self.min_cost_state is None

                if self.options['recall_when_annealing']:
                    self.reset_state()
                self.decrease_learning_rate()
                if self.options['reset_when_annealing']:
                    self.optimizer.reset()

    def validations_since_min_cost(self):
        """Returns the number of times the validation set cost has been computed
        since the minimum cost was obtained.

        :rtype: int
        :returns: number of validations since the minimum cost (0 means the last
                  validation is the best so far)
        """

        averaged_cost_history = \
            [numpy.mean(numpy.asarray(self._cost_history[i - 3:i]))
             for i in range(3, len(self._cost_history) + 1)]
        logging.debug("[%d] Cost history averaged over 3 consecutive validations:",
                      self.update_number)
        logging.debug(str(numpy.asarray(averaged_cost_history)))

        if len(averaged_cost_history) == 0:
            return -1
        else:
            return numpy.argmin(averaged_cost_history[::-1])
