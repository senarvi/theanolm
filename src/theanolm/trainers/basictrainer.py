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
        # state of minimum cost found so far
        self.min_cost_state = None
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
            self.epoch_start_ppl = \
                self.scorer.compute_perplexity(self.validation_iter)
            print("Validation set perplexity at the start of epoch {}: {}"
                  "".format(self.epoch_number,
                            self.epoch_start_ppl))

            for word_ids, _, mask in self.training_iter:
                self.update_number += 1
                self.total_updates += 1

                self.optimizer.update_minibatch(word_ids, mask)

                if (self.log_update_interval >= 1) and \
                   (self.total_updates % self.log_update_interval == 0):
                    self.log_update()

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
        perplexity = self.scorer.compute_perplexity(self.validation_iter)
        self._append_validation_cost(perplexity)
        if self.validations_since_min_cost() == 0:
            self._set_min_cost_state()

    def log_update(self):
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
        """Sets the values of Theano shared variables. If ``state`` is not
        given, uses the state of minimum validation cost found so far. If
        ``state`` is given, uses it and saves it as the minimum cost state.
        
        Requires that ``state`` contains values for all the training parameters.

        :type state: dict of numpy types
        :param state: if a dictionary of training parameters is given, takes the
                      new values from this dictionary, and assumes this is the
                      state of minimum cost found so far
        """

        if state is None:
            state = self.min_cost_state
        else:
            self.min_cost_state = state

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
        saved_cost_history = state['trainer.cost_history'].tolist()
        # If the error history was empty when the state was saved,
        # ndarray.tolist() will return None.
        if saved_cost_history is None:
            self._cost_history = []
        else:
            self._cost_history = saved_cost_history
        logging.debug("[%d] Validation set cost history since learning rate "
                      "was decreased:", self.update_number)
        logging.debug(str(numpy.asarray(self._cost_history)))

        self.optimizer.set_state(state)
        self.stopper.set_state(state)

    def decrease_learning_rate(self):
        """Called when the validation set cost stops decreasing.
        """

        self.stopper.improvement_ceased()
        self.optimizer.decrease_learning_rate()
        self._cost_history = []

    def num_validations(self):
        """Returns the number of validations since learning rate was decreased.

        :rtype: int
        :returns: size of cost history
        """

        return len(self._cost_history)

    def has_improved(self):
        """Tests whether validation set cost has decreased enough after the
        most recent mini-batch update.

        TODO: Implement a test for statistical significance.

        :rtype: bool
        :returns: True if validation set cost has decreased enough, False
                  otherwise
        """

        if len(self._cost_history) == 0:
            raise RuntimeError("BasicTrainer.cost_has_improved() "
                               "called with empty cost history.")
        else:
            return self._cost_history[-1] < \
                   0.999 * min(self._cost_history[:-1])

    def validations_since_min_cost(self):
        """Returns the number of times the validation set cost has been computed
        since the minimum cost was obtained.

        :rtype: int
        :returns: number of validations since the minimum cost (0 means the last
                  validation is the best so far)
        """

        if len(self._cost_history) == 0:
            raise RuntimeError("BasicTrainer.validations_since_min_cost() "
                               "called with empty cost history.")
        else:
            # Reverse the order of self._cost_history to find the last element
            # with the minimum value (in case there are several elements with the
            # same value.
            return numpy.argmin(self._cost_history[::-1])

    def _append_validation_cost(self, validation_cost):
        """Adds the validation set cost to the cost history.

        :type validation_cost: float
        :param validation_cost: the new validation set cost to be added to the history
        """

        self._cost_history.append(validation_cost)
        logging.debug("[%d] Validation set cost history since learning rate was "
                      "decreased:", self.update_number)
        logging.debug(str(numpy.asarray(self._cost_history)))

    def _set_min_cost_state(self, state=None):
        """Saves neural network and training state to ``self.min_cost_state``
        and writes to disk.

        :type state: dict
        :param state: if set to other than None, get the state from this
                      dictionary, instead of the current state
        """

        if state == None:
            state = self.get_state()
        self.min_cost_state = state

        path = self.model_path
        if not path is None:
            numpy.savez(path, **state)
            logging.info("Saved %d parameters to %s.", len(state), path)

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
            # This is the minimum cost so far.
            self._set_min_cost_state()
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
