#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy
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
            self.validation_state = self.get_state()

        # The rest of the function will be executed only at the final sampling
        # point.
        if not last_sample:
            return
        logging.debug("[%d] Last validation sample, perplexity %.2f.",
                      self.update_number,
                      perplexity)

        statistic = self.statistic_function(self.local_perplexities)
        self._append_validation_cost(statistic)
        logging.debug("[%d] %d samples collected, statistic %.2f.",
                      self.update_number,
                      len(self.local_perplexities),
                      statistic)

        validations_since_best = self.validations_since_min_cost()
        if validations_since_best == 0:
            # This is the minimum cost so far. Take the state at the actual
            # validation point and replace the cost history with the current
            # cost history that also includes this latest cost.
            self.validation_state['trainer.cost_history'] = \
                numpy.asarray(self._cost_history)
            self._set_min_cost_state(self.validation_state)
        elif (self.options['annealing_patience'] >= 0) and \
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
        else:
            logging.debug("%d validations since the minimum cost state.")

        self.local_perplexities = []
        self.validation_state = None
