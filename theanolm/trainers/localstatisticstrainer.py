#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy
import h5py
from theanolm.trainers.basictrainer import BasicTrainer

class LocalStatisticsTrainer(BasicTrainer):
    """A trainer that samples perplexity at a number of points before the actual
    validation point, and uses statistics of the local samples as the validation
    cost. When a minimum of the cost is found, the state at the middle of the
    sampled points will be saved.
    """

    def __init__(self,
                 *args,
                 statistic_function=lambda x: numpy.median(numpy.asarray(x)),
                 **kwargs):
        """Creates empty member variables for the perplexities list and the
        training state at the validation point. Training state is saved at only
        one validation point at a time, so validation interval is at least the
        the number of samples used per validation.

        :type statistic_function: Python function
        :param statistic_function: a function to be performed on a list to
                                   collect the statistic to be used as a cost
                                   (median by default)
        """

        super().__init__(*args, **kwargs)

        self.samples_per_validation = 10
        self.local_perplexities = []
        self.statistic_function = statistic_function
        self.validation_state = None

    def _validate(self, perplexity):
        """If at or just before the actual validation point, computes perplexity
        and adds to the list of samples. After the actual validation point,
        computes new perplexity values until samples_per_validation values have
        been computed.

        :type perplexity: float
        :param perplexity: computed perplexity at a validation point, None
                           elsewhere
        """

        if not self._is_scheduled(self.options['validation_frequency'],
                                  self.samples_per_validation - 1):
            return

        # Perplexity is set at the actual validation point, which is the final
        # sampling point. Otherwise we'll compute it here.
        last_sample = True
        if perplexity is None:
            perplexity = self.scorer.compute_perplexity(self.validation_iter)
            last_sample = False
        
        self.local_perplexities.append(perplexity)
        if len(self.local_perplexities) == 1:
            logging.debug("[%d] First validation sample, perplexity %.2f.",
                          self.update_number,
                          perplexity)
    
        if not self._is_scheduled(self.options['validation_frequency'],
                                  self.samples_per_validation // 2):
            return

        # The first sampling point within samples_per_validation / 2 of the
        # actual validation point is the center of the sampling points.
        if self.validation_state is None:
            logging.debug("[%d] Center of validation, perplexity %.2f.",
                          self.update_number,
                          perplexity)
            self.validation_state = h5py.File(
                name='validation-state', driver='core', backing_store=False)
            self.get_state(self.validation_state)

        # The rest of the function will be executed only at the final sampling
        # point.
        if not last_sample:
            return
        logging.debug("[%d] Last validation sample, perplexity %.2f.",
                      self.update_number,
                      perplexity)

        if len(self.local_perplexities) < self.samples_per_validation:
            # After restoring a previous validation state, which is at the
            # center of the sampling points, the trainer will collect again half
            # of the samples. Don't take that as a validation.
            logging.debug("[%d] Only %d samples collected. Ignoring this "
                          "validation.",
                          self.update_number,
                          len(self.local_perplexities))
            self.local_perplexities = []
            self.validation_state.close()
            self.validation_state = None
            return

        statistic = self.statistic_function(self.local_perplexities)
        self._cost_history = numpy.append(self._cost_history, statistic)
        if self._has_improved():
            # Take the state at the actual validation point and replace the cost
            # history with the current cost history that also includes this
            # latest statistic.
            h5_cost_history = self.validation_state['trainer/cost_history']
            h5_cost_history.resize(self._cost_history.shape)
            h5_cost_history[:] = self._cost_history
            self._set_candidate_state(self.validation_state)

        self._log_validation()

        if (self.options['patience'] >= 0) and \
           (self.validations_since_candidate() > self.options['patience']):
            # Too many validations without finding a new candidate state.

            # If any validations have been done, the best state has been found
            # and saved. If training has been started from previous state,
            # _candidate_state has been set to the initial state.
            assert not self._candidate_state is None

            self._decrease_learning_rate()

        self.local_perplexities = []
        self.validation_state.close()
        self.validation_state = None
