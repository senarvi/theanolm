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
        """

        super().__init__(*args, **kwargs)

        self.samples_per_validation = 10
        self.local_perplexities = []
        self.stat_function = stat_function
        self.network_state_validation = None
        self.trainer_state_validation = None
        self.validation_update_number = None

    def _validate(self, perplexity):
        """If at or just before the actual validation point, computes perplexity
        and adds to the list of samples. After the actual validation point,
        computes new perplexity values until samples_per_validation values have
        been computed.
        """

        if not perplexity is None:
            # the actual validation point
            self.network_state_validation = self.network.get_state()
            self.trainer_state_validation = self.get_state()
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
        if len(self.local_perplexities) == 0:
            logging.debug("First sample collected at update %d, perplexity %.2f.",
                          self.update_number,
                          perplexity)
        self.local_perplexities.append(perplexity)
        if len(self.local_perplexities) < self.samples_per_validation:
            return

        assert not self.network_state_validation is None
        assert not self.trainer_state_validation is None

        stat = self.stat_function(self.local_perplexities)
        self._append_validation_cost(stat)
        logging.debug("%d samples collected at update %d, validation center at %d, stat %.2f.",
                      len(self.local_perplexities),
                      self.update_number,
                      self.validation_update_number,
                      stat)
        self.local_perplexities = []

        validations_since_best = self._validations_since_min_cost()
        if validations_since_best == 0:
            # This is the minimum cost so far.
            self.network_state_min_cost = self.network_state_validation
            self.network_state_validation = None
            self.trainer_state_min_cost = self.trainer_state_validation
            self.trainer_state_validation = None
            self.save_model()
        elif (self.options['wait_improvement'] >= 0) and \
             (validations_since_best > self.options['wait_improvement']):
            # Too many validations without improvement.
            if self.options['recall_when_annealing']:
                self.network.set_state(self.network_state_min_cost)
                self.set_state(self.trainer_state_min_cost)
            self.decrease_learning_rate()
            if self.options['reset_when_annealing']:
                self.optimizer.reset()
