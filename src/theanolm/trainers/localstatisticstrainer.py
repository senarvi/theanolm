#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy
from theanolm.trainers.basictrainer import BasicTrainer

class LocalStatisticsTrainer(BasicTrainer):
    """A trainer that samples perplexity at several points before and after the
    actual validation point, and uses statistics of the local samples as the
    validation cost.
    """

    def __init__(self,
                 *args,
                 stat_function=lambda x: numpy.median(numpy.asarray(x)),
                 **kwargs):
        """Creates empty member variables for the perplexities list and the
        training state at the validation point. Training state is saved at only
        one validation point at a time, so validation interval is at least the
        the number of samples used per validation.

        :type stat_function: Python function
        :param stat_function: a function to be performed on a list to collect
                              the statistic to be used as a cost (median by
                              default)
        """

        super().__init__(*args, **kwargs)

        self.samples_per_validation = 10
        self.local_perplexities = []
        self.stat_function = stat_function
        self.validation_state = None
        self.validation_update_number = None

    def _validate(self, perplexity):
        """If at or just before the actual validation point, computes perplexity
        and adds to the list of samples. After the actual validation point,
        computes new perplexity values until samples_per_validation values have
        been computed.

        :type perplexity: float
        :param perplexity: computed perplexity at a validation point, None
                           elsewhere
        """

        if not perplexity is None:
            # the actual validation point
            self.validation_state = self.get_state()
            self.validation_update_number = self.update_number
            self.__add_sample(perplexity)

        elif self._is_scheduled(self.options['validation_frequency'],
                                self.samples_per_validation // 2):
            # within samples_per_validation / 2 updates before the validation
            # point
            perplexity = self.scorer.compute_perplexity(self.validation_iter)
            self.__add_sample(perplexity)

        elif len(self.local_perplexities) > 0:
            # after the actual validation point, before samples_per_validation
            # perplexities have been computed
            perplexity = self.scorer.compute_perplexity(self.validation_iter)
            self.__add_sample(perplexity)

    def __add_sample(self, perplexity):
        """Appends computed perplexity to the list of local samples, and if
        enough samples have been collected, computes the statistic, appends to
        the cost history, and validates whether there was improvement.

        :type perplexity: float
        :param perplexity: computed perplexity
        """

        self.local_perplexities.append(perplexity)
        if len(self.local_perplexities) < self.samples_per_validation:
            if len(self.local_perplexities) == 1:
                logging.debug("[%d] First sample collected, perplexity %.2f.",
                              self.update_number,
                              perplexity)
            return

        # Sampling is started samples_per_validation / 2 updates before the
        # validation point, so now we have passed the validation point.
        assert not self.validation_state is None

        stat = self.stat_function(self.local_perplexities)
        self._append_validation_cost(stat)
        logging.debug("[%d] %d samples collected, validation center at %d, "
                      "stat %.2f.",
                      self.update_number,
                      len(self.local_perplexities),
                      self.validation_update_number,
                      stat)
        self.local_perplexities = []

        validations_since_best = self.validations_since_min_cost()
        if validations_since_best == 0:
            # This is the minimum cost so far. Take the state at the actual
            # validation point and replace the cost history with the current
            # cost history that also includes this latest cost.
            self.validation_state['trainer.cost_history'] = \
                numpy.asarray(self._cost_history)
            self._set_min_cost_state(self.validation_state)
            self.validation_state = None
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
