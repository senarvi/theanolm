#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from theanolm.training.basictrainer import BasicTrainer

class ValidationAverageTrainer(BasicTrainer):
    """A trainer that computes the average of the perplexity from three
    consecutive validations.
    """

    def __init__(self, dictionary, network, scorer,
                 training_file, validation_iter,
                 initial_state, training_options, optimization_options,
                 profile=False):
        super().__init__(dictionary, network, scorer,
                         training_file, validation_iter,
                         initial_state, training_options, optimization_options,
                         profile)
        self.network_state_previous = None
        self.optimizer_state_previous = None

    def _validate(self, perplexity):
        if perplexity is None:
            return

        self._append_validation_cost(perplexity)

        validations_since_best = self._validations_since_min_cost()
        if validations_since_best == 0:
            # The minimum cost is from the three previous validations.
            # Previous validation is in the middle of those validations.
            self.network_state_min_cost = self.network_state_previous
            self.trainer_state_min_cost = self.trainer_state_previous
            self.save_model()
        else:
            self.network_state_previous = self.network.get_state()
            self.trainer_state_previous = self.get_state()
            if (self.options['wait_improvement'] >= 0) and \
               (validations_since_best > self.options['wait_improvement']):
                logging.debug("%d validations since the minimum perplexity was "
                              "measured. Decreasing learning rate.",
                              validations_since_best)
                if self.options['recall_when_annealing']:
                    self.network.set_state(self.network_state_min_cost)
                    self.set_state(self.trainer_state_min_cost)
                self.decrease_learning_rate()
                if self.options['reset_when_annealing']:
                    self.optimizer.reset()

    def _validations_since_min_cost(self):
        """Returns the number of times the validation set cost has been computed
        since the minimum cost was obtained.

        :rtype: int
        :returns: number of validations since the minimum cost (0 means the last
                  validation is the best so far)
        """

        averaged_cost_history = \
            [numpy.mean(numpy.asarray(self._cost_history[i - 3:i]))
             for i in range(3, len(self._cost_history) + 1)]
        logging.debug("Cost history averaged over 3 consecutive validations:")
        logging.debug(str(numpy.asarray(averaged_cost_history)))

        if len(averaged_cost_history) == 0:
            return -1
        else:
            return numpy.argmin(averaged_cost_history[::-1])
