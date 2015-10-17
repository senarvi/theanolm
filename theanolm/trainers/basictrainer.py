#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import logging
import mmap
import numpy
from theanolm import find_sentence_starts, ShufflingBatchIterator
from theanolm.exceptions import IncompatibleStateError, NumberError
from theanolm.optimizers import create_optimizer
from theanolm.stoppers import create_stopper

class BasicTrainer(object):
    """Basic training process saves a history of validation costs and "
    decreases learning rate when the cost does not decrease anymore.
    """

    def __init__(self, training_options, optimization_options,
                 network, dictionary, scorer,
                 training_file, validation_iter,
                 profile=False):
        """Creates the optimizer and initializes the training process.

        :type dictionary: theanolm.Dictionary
        :param dictionary: dictionary that provides mapping between words and
                           word IDs

        :type network: theanolm.Network
        :param network: a neural network to be trained

        :type scorer: theanolm.TextScorer
        :param scorer: a text scorer for computing validation set perplexity

        :type validation_iter: theanolm.BatchIterator
        :param validation_iter: an iterator for computing validation set
                                perplexity

        :type training_options: dict
        :param training_options: a dictionary of training options

        :type optimization_options: dict
        :param optimization_options: a dictionary of optimization options
        """

        self.network = network
        self.scorer = scorer
        self.validation_iter = validation_iter

        self.optimizer = create_optimizer(optimization_options,
                                          self.network,
                                          profile)

        training_mmap = mmap.mmap(training_file.fileno(),
                                  0,
                                  prot=mmap.PROT_READ)

        print("Finding sentence start positions in training data.")
        sys.stdout.flush()
        sentence_starts = find_sentence_starts(training_mmap)

        self.training_iter = ShufflingBatchIterator(
            training_mmap,
            dictionary,
            sentence_starts,
            batch_size=training_options['batch_size'],
            max_sequence_length=training_options['sequence_length'])

        print("Computing the number of training updates per epoch.")
        sys.stdout.flush()
        self.updates_per_epoch = len(self.training_iter)

        self.stopper = create_stopper(training_options, self)
        self.options = training_options

        # path where the model and training state will be saved
        self.model_path = None
        # current candidate for the minimum validation cost state
        self._candidate_state = None
        # index to the cost history that corresponds to the current candidate
        # state
        self._candidate_index = None
        # number of mini-batch updates between log messages
        self.log_update_interval = 0
        # current training epoch
        self.epoch_number = 1
        # number of mini-batch updates performed in this epoch
        self.update_number = 0
        # total number of mini-batch updates performed (after restart)
        self.total_updates = 0
        # validation set cost history
        self._cost_history = []

    def set_model_path(self, path):
        self.model_path = path

    def set_logging(self, interval):
        self.log_update_interval = interval

    def run(self):
        while self.stopper.start_new_epoch():
            for word_ids, _, mask in self.training_iter:
                self.update_number += 1
                self.total_updates += 1

                self.optimizer.update_minibatch(word_ids, mask)

                if (self.log_update_interval >= 1) and \
                   (self.total_updates % self.log_update_interval == 0):
                    self._log_update()

                if self._is_scheduled(self.options['validation_frequency']):
                    perplexity = self.scorer.compute_perplexity(self.validation_iter)
                    if numpy.isnan(perplexity) or numpy.isinf(perplexity):
                        raise NumberError("Validation set perplexity computation resulted "
                                          "in a numerical error.")
                else:
                    perplexity = None
                self._validate(perplexity)

                if not self.stopper.start_new_minibatch():
                    break

            self.epoch_number += 1
            self.update_number = 0

        logging.info("Training finished.")

    def get_state(self):
        """Pulls parameter values from Theano shared variables and returns a
        dictionary of all the network and training state variables.

        For consistency, all the parameter values are returned as numpy types,
        since state read from a model file also contains numpy types. This also
        ensures the cost history will be copied into the returned dictionary.

        :rtype: dict of numpy types
        :returns: a dictionary of the parameter values
        """

        result = self.network.get_state()
        result['trainer.epoch_number'] = numpy.int64(self.epoch_number)
        result['trainer.update_number'] = numpy.int64(self.update_number)
        result['trainer.cost_history'] = numpy.array(self._cost_history)
        result.update(self.optimizer.get_state())
        result.update(self.stopper.get_state())
        return result

    def reset_state(self, state=None):
        """Resets the values of Theano shared variables to a state that gives a
        minimum of the validation set cost. If ``state`` is not given, uses the
        current candidate state. If ``state`` is given, uses it and saves it as
        the new candidate state.

        Sets candidate state index point to the last element in the loaded cost
        history.
        
        Requires that ``state`` contains values for all the training parameters.

        :type state: dict of numpy types
        :param state: if a dictionary of training parameters is given, takes the
                      new values from this dictionary, and assumes this is the
                      state of minimum cost found so far
        """

        if state is None:
            state = self._candidate_state
        else:
            self._candidate_state = state

        self.network.set_state(state)

        if not 'trainer.epoch_number' in state:
            raise IncompatibleStateError("Current epoch number is missing from "
                                         "training state.")
        self.epoch_number = state['trainer.epoch_number'].item()

        if not 'trainer.update_number' in state:
            raise IncompatibleStateError("Current update number is missing "
                                         "from training state.")
        self.update_number = state['trainer.update_number'].item()
        logging.info("[%d] (%.2f %%) of epoch %d",
                     self.update_number,
                     self.update_number / self.updates_per_epoch * 100,
                     self.epoch_number)

        if not 'trainer.cost_history' in state:
            raise IncompatibleStateError("Validation set cost history is "
                                         "missing from training state.")
        self._cost_history = state['trainer.cost_history'].tolist()
        # If the cost history was empty when the state was saved,
        # ndarray.tolist() will return None.
        if self._cost_history is None:
            raise IncompatibleStateError("Validation set cost history is "
                                         "empty in the training state.")
        self._log_validation()

        self._candidate_index = len(self._cost_history) - 1

        self.optimizer.set_state(state)
        self.stopper.set_state(state)

    def decrease_learning_rate(self):
        """Called when the validation set cost stops decreasing.
        """

        logging.debug("Performance on validation set has ceased to improve.")

        self.stopper.improvement_ceased()
        self.optimizer.decrease_learning_rate()

    def num_validations(self):
        """Returns the number of validations performed.

        :rtype: int
        :returns: size of cost history
        """

        return len(self._cost_history)

    def validations_since_candidate(self):
        """Returns the number of times the validation set cost has been computed
        since the current candidate for optimal state was obtained.

        :rtype: int
        :returns: number of validations since the current candidate state (0
                  means the current candidate is the last validation)
        """

        if len(self._cost_history) == 0:
            raise RuntimeError("BasicTrainer.validations_since_candidate() "
                               "called with empty cost history.")

        return len(self._cost_history) - 1 - self._candidate_index

    def candidate_cost(self):
        """Returns the validation set cost given by the current candidate for
        the minimum cost state.

        :rtype: float
        :returns: current candidate state cost, or None if the candidate state
                  has not been set yet
        """

        if self._candidate_index is None:
            return None

        return self._cost_history[self._candidate_index]

    def _has_improved(self):
        """Tests whether the previously computed validation set cost was
        significantly better than the cost given by the current candidate state.

        TODO: Implement a test for statistical significance.

        :rtype: bool
        :returns: True if validation set cost decreased enough, or there was no
                  previous candidate state; False otherwise
        """

        if len(self._cost_history) == 0:
            raise RuntimeError("BasicTrainer._has_improved() called with empty "
                               "cost history.")

        if self._candidate_index is None:
            return True

        return self._cost_history[-1] < 0.999 * self.candidate_cost()

    def _log_update(self):
        """Logs information about the previous mini-batch update.
        """

        logging.info("[%d] (%.2f %%) of epoch %d -- lr = %g, cost = %.2f, "
                     "duration = %.2f ms",
                     self.update_number,
                     self.update_number / self.updates_per_epoch * 100,
                     self.epoch_number,
                     self.optimizer.get_learning_rate(),
                     self.optimizer.update_cost,
                     self.optimizer.update_duration * 100)

    def _log_validation(self):
        """Prints the validation set cost history (or its tail), highlighting
        the candidate for the minimum cost.
        """

        str_costs = ["%.1f" % x for x in self._cost_history]
        if not self._candidate_index is None:
            str_costs[self._candidate_index] = \
                '[' + str_costs[self._candidate_index] + ']'
        logging.debug("[%d] Validation set cost history: %s",
                      self.update_number,
                      ' '.join(str_costs[-20:]))

    def _set_candidate_state(self, state=None, index=None):
        """Sets neural network and training state as the candidate for the
        minimum validation cost state, and writes to disk.

        :type state: dict
        :param state: if set to a dictionary, read the state from the dictionary
                      items, instead of the current state

        :type index: int
        :param index: index to the cost history that points to the candidate
                      state, or None for the last item of the cost history
        """

        if state is None:
            state = self.get_state()
        if index is None:
            index = len(self._cost_history) - 1

        self._candidate_state = state
        self._candidate_index = index

        path = self.model_path
        if not path is None:
            numpy.savez(path, **state)
            logging.info("New candidate for optimal state. Saved %d parameters "
                         "to %s.", len(state), path)
        else:
            logging.debug("New candidate for optimal state.")

    def _validate(self, perplexity):
        """When ``perplexity`` is not None, appends it to cost history and
        validates whether there was improvement.

        :type perplexity: float
        :param perplexity: computed perplexity at a validation point, None
                           elsewhere
        """

        if perplexity is None:
            return

        self._cost_history.append(perplexity)

        if self._has_improved():
            self._set_candidate_state()

        self._log_validation()

        if (self.options['patience'] >= 0) and \
           (self.validations_since_candidate() > self.options['patience']):
            # Too many validations without finding a new candidate state.

            # If any validations have been done, the best state has been found
            # and saved. If training has been started from previous state,
            # _candidate_state has been set to the initial state.
            assert not self._candidate_state is None

            self.reset_state()
            self.decrease_learning_rate()
            if self.options['reset_when_annealing']:
                self.optimizer.reset()


    def _is_scheduled(self, frequency, within=0):
        """Checks if an event is scheduled to be performed within given number
        of updates after this point.

        For example, updates_per_epoch=9, frequency=2:

        update_number:  1   2   3   4  [5]  6   7   8  [9] 10  11  12
        * frequency:    2   4   6   8  10  12  14  16  18  20  22  24
        modulo:         2   4   6   8   1   3   5   7   0   2   4   6
        within:         4   3   2   1   0   3   2   1   0   4   3   2
        * frequency:    8   6   4   2   0   6   4   2   0   8   6   4

        :type frequency: int
        :param frequency: how many times per epoch the event should be
                          performed

        :type within: int
        :param within: if zero, returns True if the event should be performed
                       now; otherwise returns True if the event should be
                       performed within this many updates in the future

        :rtype: bool
        :returns: whether the operation is scheduled to be performed
        """

        modulo = self.update_number * frequency % self.updates_per_epoch
        return modulo < frequency or \
               self.updates_per_epoch - modulo <= within * frequency
