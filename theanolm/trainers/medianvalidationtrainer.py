#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
from theanolm.trainers.basictrainer import BasicTrainer

class MedianValidationTrainer(BasicTrainer):
    """A trainer that computes perplexity at several points ahead of the actual
    validation point and computes the median.

    Validating at a single training point is sensitive to noise in the computed
    perplexity values. Median validation allows faster reaction to increasing
    validation set perplexity.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples_per_validation = 10
        self.current_score = []

    def _validate(self, perplexity):
        """When approaching a validation point, computes perplexity and appends
        it to cost history. At a validation point computes median and validates
        whether there was improvement.

        :type perplexity: float
        :param perplexity: computed perplexity at a validation point, None
                           elsewhere
        """

        if not perplexity is None:
            self.current_score.append(perplexity)
            median = numpy.median(self.current_score)
            print("Validation set perplexity at minibatch {}: {} "
                  "(median of {}, std {}".format(self.total_updates,
                                                 self.current_score,
                                                 len(self.current_score),
                                                 numpy.std(self.current_score)))
            super()._validate(median)

        elif self._is_scheduled(self.options['validation_frequency'],
                                self.samples_per_validation - 1):
            perplexity = self.scorer.compute_perplexity(self.validation_iter)
            self.current_score.append(perplexity)
            if len(self.current_score) > self.samples_per_validation:
                self.current_score.pop(0)
