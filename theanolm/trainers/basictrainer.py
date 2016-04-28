#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import logging
import mmap
import numpy
import theano
from theanolm import ShufflingBatchIterator
from theanolm.exceptions import IncompatibleStateError, NumberError
from theanolm.optimizers import create_optimizer
from theanolm.stoppers import create_stopper

class BasicTrainer(object):
    """Basic training process saves a history of validation costs and "
    decreases learning rate when the cost does not decrease anymore.
    """

    def __init__(self, training_options, optimization_options,
                 network, vocabulary, scorer,
                 training_files, sampling, validation_iter, state,
                 profile=False):
        """Creates the optimizer and initializes the training process.

        :type training_options: dict
        :param training_options: a dictionary of training options

        :type optimization_options: dict
        :param optimization_options: a dictionary of optimization options

        :type network: theanolm.Network
        :param network: a neural network to be trained

        :type vocabulary: theanolm.Vocabulary
        :param vocabulary: vocabulary that provides mapping between words and
                           word IDs

        :type scorer: theanolm.TextScorer
        :param scorer: a text scorer for computing validation set perplexity

        :type training_files: list of file objects
        :param training_files: list of files to be used as training data

        :type sampling: list of floats
        :param sampling: specifies a fraction for each training file, how much
                         to sample on each epoch

        :type validation_iter: theanolm.BatchIterator
        :param validation_iter: an iterator for computing validation set
                                perplexity

        :type state: h5py.File
        :param state: HDF5 file where initial training state will be possibly
                      read from, and candidate states will be saved to

        :type profile: bool
        :param profile: if set to True, creates Theano profile objects
        """

        self.network = network
        self.vocabulary = vocabulary
        self.scorer = scorer
        self.validation_iter = validation_iter

        self.optimizer = create_optimizer(optimization_options,
                                          self.network,
                                          profile)

        self.training_iter = ShufflingBatchIterator(
            training_files,
            sampling,
            vocabulary,
            batch_size=training_options['batch_size'],
            max_sequence_length=training_options['sequence_length'])

        print("Computing the number of training updates per epoch.")
        sys.stdout.flush()
        self.updates_per_epoch = len(self.training_iter)
        if self.updates_per_epoch < 1:
            raise ValueError("Training data does not contain any sentences.")

        self.stopper = create_stopper(training_options, self)
        self.options = training_options

        # current candidate for the minimum validation cost state
        self._candidate_state = state
        if 'trainer' in self._candidate_state:
            print("Restoring initial network state from {}.".format(
                self._candidate_state.filename))
            sys.stdout.flush()
            self._reset_state()
        else:
            # index to the cost history that corresponds to the current candidate
            # state
            self._candidate_index = None
            # current training epoch
            self.epoch_number = 1
            # number of mini-batch updates performed in this epoch
            self.update_number = 0
            # validation set cost history
            self._cost_history = numpy.asarray([], dtype=theano.config.floatX)

        # number of mini-batch updates between log messages
        self.log_update_interval = 0
        # total number of mini-batch updates performed (after restart)
        self.total_updates = 0

    def set_logging(self, interval):
        self.log_update_interval = interval

    def run(self):
        while self.stopper.start_new_epoch():
            for word_ids, file_ids, mask in self.training_iter:
                self.update_number += 1
                self.total_updates += 1

                class_ids = self.vocabulary.word_id_to_class_id[word_ids]
                self.optimizer.update_minibatch(word_ids, class_ids, file_ids, mask)

                if (self.log_update_interval >= 1) and \
                   (self.total_updates % self.log_update_interval == 0):
                    self._log_update()

                if self._is_scheduled(self.options['validation_frequency']):
                    perplexity = self.scorer.compute_perplexity(self.validation_iter)
                    if numpy.isnan(perplexity) or numpy.isinf(perplexity):
                        raise NumberError(
                            "Validation set perplexity computation resulted "
                            "in a numerical error.")
                else:
                    perplexity = None
                self._validate(perplexity)

                if not self.stopper.start_new_minibatch():
                    break

            message = "Finished training epoch {}.".format(self.epoch_number)
            best_cost = self.candidate_cost()
            if not best_cost is None:
                message += " Best validation perplexity {:.2f}.".format(
                    best_cost)
            print(message)

            self.epoch_number += 1
            self.update_number = 0

        print("Training finished.")

    def get_state(self, state):
        """Pulls parameter values from Theano shared variables and updates a
        HDF5 file with all the network and training state variables.

        For consistency, all the parameter values are returned as numpy types,
        since state read from a model file also contains numpy types. This also
        ensures the cost history will be copied into the returned dictionary.

        :type state: h5py.File
        :param state: HDF5 file for storing the current state
        """

        h5_trainer = state.require_group('trainer')
        h5_trainer.attrs['epoch_number'] = self.epoch_number
        h5_trainer.attrs['update_number'] = self.update_number
        if 'cost_history' in h5_trainer:
            h5_trainer['cost_history'].resize(self._cost_history.shape)
            h5_trainer['cost_history'][:] = self._cost_history
        else:
            h5_trainer.create_dataset(
                'cost_history', data=self._cost_history, maxshape=(None,),
                chunks=(1000,))

        self.network.get_state(state)
        self.training_iter.get_state(state)
        self.optimizer.get_state(state)

    def _reset_state(self):
        """Resets the values of Theano shared variables to the current candidate
         state.

        Sets candidate state index point to the last element in the loaded cost
        history.
        
        Requires that if ``state`` is set, it contains values for all the
        training parameters.

        :type state: h5py.File
        :param state: if a HDF5 file is given, reads the the training parameters
                      from this file, and assumes this is the state of minimum
                      cost found so far
        """

        self.network.set_state(self._candidate_state)

        if not 'trainer' in self._candidate_state:
            raise IncompatibleStateError("Training state is missing.")
        h5_trainer = self._candidate_state['trainer']

        if not 'epoch_number' in h5_trainer.attrs:
            raise IncompatibleStateError("Current epoch number is missing from "
                                         "training state.")
        self.epoch_number = int(h5_trainer.attrs['epoch_number'])

        if not 'update_number' in h5_trainer.attrs:
            raise IncompatibleStateError("Current update number is missing from "
                                         "training state.")
        self.update_number = int(h5_trainer.attrs['update_number'])

        logging.info("[%d] (%.2f %%) of epoch %d",
                     self.update_number,
                     self.update_number / self.updates_per_epoch * 100,
                     self.epoch_number)

        if not 'cost_history' in h5_trainer:
            raise IncompatibleStateError("Validation set cost history is "
                                         "missing from training state.")
        self._cost_history = h5_trainer['cost_history'].value
        if self._cost_history.size == 0:
            raise IncompatibleStateError("Validation set cost history is "
                                         "empty in the training state.")
        self._candidate_index = self._cost_history.size - 1
        self._log_validation()

        self.training_iter.set_state(self._candidate_state)
        self.optimizer.set_state(self._candidate_state)

    def num_validations(self):
        """Returns the number of validations performed.

        :rtype: int
        :returns: size of cost history
        """

        return self._cost_history.size

    def validations_since_candidate(self):
        """Returns the number of times the validation set cost has been computed
        since the current candidate for optimal state was obtained.

        :rtype: int
        :returns: number of validations since the current candidate state (0
                  means the current candidate is the last validation)
        """

        if self._cost_history.size == 0:
            raise RuntimeError("BasicTrainer.validations_since_candidate() "
                               "called with empty cost history.")

        return self._cost_history.size - 1 - self._candidate_index

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

    def _decrease_learning_rate(self):
        """Called when the validation set cost stops decreasing.
        """

        # Current learning rate might be smaller than the one stored in the
        # state, so set the new value after restoring optimizer to the old
        # state.
        old_value = self.optimizer.learning_rate
        new_value = old_value / 2
        self._reset_state()
        self.stopper.improvement_ceased()
        self.optimizer.learning_rate = new_value

        print("Model performance stopped improving. Decreasing learning rate "
              "from {} to {} and resetting state to {:.0f} % of epoch {}."
              .format(old_value,
                      new_value,
                      self.update_number / self.updates_per_epoch * 100,
                      self.epoch_number))

    def _has_improved(self):
        """Tests whether the previously computed validation set cost was
        significantly better than the cost given by the current candidate state.

        TODO: Implement a test for statistical significance.

        :rtype: bool
        :returns: True if validation set cost decreased enough, or there was no
                  previous candidate state; False otherwise
        """

        if self._cost_history.size == 0:
            raise RuntimeError("BasicTrainer._has_improved() called with empty "
                               "cost history.")

        if self._candidate_index is None:
            return True

        return self._cost_history[-1] < 0.999 * self.candidate_cost()

    def _log_update(self):
        """Logs information about the previous mini-batch update.
        """

        logging.info("[%d] (%.1f %%) of epoch %d -- lr = %.1g, cost = %.2f, "
                     "duration = %.1f ms",
                     self.update_number,
                     self.update_number / self.updates_per_epoch * 100,
                     self.epoch_number,
                     self.optimizer.learning_rate,
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

    def _set_candidate_state(self, state=None):
        """Sets neural network and training state as the candidate for the
        minimum validation cost state, and writes to disk.

        :type state: h5py.File
        :param state: if a HDF5 file is given, reads the state from this file,
                      instead of the current state
        """

        if state is None:
            self.get_state(self._candidate_state)
        else:
            state.flush()
            for name in state:
                if name in self._candidate_state:
                    del self._candidate_state[name]
                self._candidate_state.copy(state[name], name, expand_refs=True)
            for name in state.attrs:
                self._candidate_state.attrs[name] = state.attrs[name]

        self._candidate_index = self._cost_history.size - 1

        self._candidate_state.flush()
        logging.info("New candidate for optimal state saved to %s.",
                     self._candidate_state.filename)

    def _validate(self, perplexity):
        """When ``perplexity`` is not None, appends it to cost history and
        validates whether there was improvement.

        :type perplexity: float
        :param perplexity: computed perplexity at a validation point, None
                           elsewhere
        """

        if perplexity is None:
            return

        self._cost_history = numpy.append(self._cost_history, perplexity)

        if self._has_improved():
            self._set_candidate_state()

        self._log_validation()

        if (self.options['patience'] >= 0) and \
           (self.validations_since_candidate() > self.options['patience']):
            # Too many validations without finding a new candidate state.

            # If any validations have been done, the best state has been found
            # and saved. If training has been started from previous state,
            # _candidate_state has been set to the initial state.
            assert self._candidate_state.keys()

            self._decrease_learning_rate()


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
