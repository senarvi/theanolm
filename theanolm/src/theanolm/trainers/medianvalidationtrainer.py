#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        self.num_validations = 10
        self.current_score = []

    def _validate(self, perplexity):
        if not perplexity is None:
            self.current_score.append(perplexity)
            median = numpy.median(self.current_score)
            print("Validation set perplexity at minibatch {}: {} "
                  "(median of {}, std {}".format(self.total_updates,
                                                 self.current_score,
                                                 len(self.current_score),
                                                 numpy.std(self.current_score)))
            super()._validate(median)

        elif self._updates_to_next_event(self.options['validation_frequency']) \
             < self.num_validations:
            perplexity = self.scorer.compute_perplexity(self.validation_iter)
            self.current_score.append(perplexity)
            if len(self.current_score) > self.num_validations:
                self.current_score.pop(0)
