#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import logging
from time import time
import h5py
import numpy
import theano
from theanolm import ShufflingBatchIterator, LinearBatchIterator
from theanolm.exceptions import IncompatibleStateError, NumberError
from theanolm.training.stoppers import create_stopper

class Trainer(object):
    """Training Process
    
    Saves a history of validation costs and decreases learning rate when the
    cost does not decrease anymore.
    """

    def __init__(self, training_options, vocabulary, training_files, sampling):
        """Creates the optimizer and initializes the training process.

        Creates empty member variables for the perplexities list and the
        training state at the validation point. Training state is saved at only
        one validation point at a time, so validation interval is at least the
        the number of samples used per validation.

        :type training_options: dict
        :param training_options: a dictionary of training options

        :type vocabulary: Vocabulary
        :param vocabulary: vocabulary that provides mapping between words and
                           word IDs

        :type training_files: list of file objects
        :param training_files: list of files to be used as training data

        :type sampling: list of floats
        :param sampling: specifies a fraction for each training file, how much
                         to sample on each epoch
        """

        self._vocabulary = vocabulary

        print("Computing unigram probabilities and the number of mini-batches "
              "in training data.")
        linear_iter = LinearBatchIterator(
            training_files,
            vocabulary,
            batch_size=training_options['batch_size'],
            max_sequence_length=training_options['sequence_length'])
        sys.stdout.flush()
        self._updates_per_epoch = 0
        class_counts = numpy.zeros(vocabulary.num_classes(), dtype='int64')
        for word_ids, _, mask in linear_iter:
            self._updates_per_epoch += 1
            word_ids = word_ids[mask == 1]
            class_ids = vocabulary.word_id_to_class_id[word_ids]
            numpy.add.at(class_counts, class_ids, 1)
        if self._updates_per_epoch < 1:
            raise ValueError("Training data does not contain any sentences.")
        logging.debug("One epoch of training data contains %d mini-batch updates.",
                      self._updates_per_epoch)
        self.class_prior_probs = class_counts / class_counts.sum()
        logging.debug("Class unigram probabilities are in the range [%.8f, "
                      "%.8f].",
                      self.class_prior_probs.min(),
                      self.class_prior_probs.max())

        self._training_iter = ShufflingBatchIterator(
            training_files,
            sampling,
            vocabulary,
            batch_size=training_options['batch_size'],
            max_sequence_length=training_options['sequence_length'])

        self._stopper = create_stopper(training_options, self)
        self._options = training_options

        # iterator to cross-validation data, or None for no cross-validation
        self._validation_iter = None
        # a text scorer for performing cross-validation
        self._scorer = None
        # number of perplexity samples per validation
        self._samples_per_validation = 7
        # function for combining validation samples
        self._statistic_function = lambda x: numpy.median(numpy.asarray(x))
        # the stored validation samples
        self._local_perplexities = []
        # the state at the center of validation samples
        self._validation_state = None

        # number of mini-batch updates between log messages
        self._log_update_interval = 0

        # the network to be trained
        self._network = None
        # the optimization function
        self._optimizer = None
        # current candidate for the minimum validation cost state
        self._candidate_state = None

    def set_validation(self, validation_iter, scorer,
                       samples_per_validation=None, statistics_function=None):
        """Sets cross-validation iterator and parameters.

        :type validation_iter: BatchIterator
        :param validation_iter: an iterator for computing validation set
                                perplexity

        :type scorer: TextScorer
        :param scorer: a text scorer for computing validation set perplexity

        :type samples_per_validation: int
        :param samples_per_validation: number of perplexity samples to compute
                                       per cross-validation

        :type statistic_function: Python function
        :param statistic_function: a function to be performed on a list of
           consecutive perplexity measurements to compute the validation cost
           (median by default)
        """

        self._validation_iter = validation_iter
        self._scorer = scorer

        if not samples_per_validation is None:
            self._samples_per_validation = samples_per_validation

        if not statistics_function is None:
            self._statistics_function = statistics_function

    def set_logging(self, log_interval):
        """Sets logging parameters.

        :type log_interval: int
        :param log_interval: number of mini-batch updates between log messages
        """

        self._log_update_interval = log_interval

    def initialize(self, network, state, optimizer):
        """Sets the network and the HDF5 file that stores the network state,
        optimizer, and validation scorer and iterator.

        If the HDF5 file contains a network state, initializes the network with
        that state.

        :type network: Network
        :param network: the network, which will be used to retrieve state when
                        saving

        :type state: h5py.File
        :param state: HDF5 file where initial training state will be possibly
                      read from, and candidate states will be saved to

        :type optimizer: BasicOptimizer
        :param optimizer: one of the optimizer implementations
        """

        self._network = network
        self._optimizer = optimizer

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

        # total number of mini-batch updates performed (after restart)
        self._total_updates = 0

    def train(self):
        """Trains a neural network.

        If cross-validation has been configured using ``set_validation()``,
        computes the validation set perplexity as many times per epoch as
        specified by the _validation_frequency_ option and saves the model when
        the perplexity improves. Otherwise saves the model after each epoch.
        """

        if (self._network is None) or (self._optimizer is None) or \
           (self._candidate_state is None):
            raise RuntimeError("Trainer has not been initialized before "
                               "calling train().")

        start_time = time()
        while self._stopper.start_new_epoch():
            epoch_start_time = time()
            for word_ids, file_ids, mask in self._training_iter:
                self.update_number += 1
                self._total_updates += 1

                class_ids = self._vocabulary.word_id_to_class_id[word_ids]
                update_start_time = time()
                self._optimizer.update_minibatch(word_ids, class_ids, file_ids, mask)
                self._update_duration = time() - update_start_time

                if (self._log_update_interval >= 1) and \
                   (self._total_updates % self._log_update_interval == 0):
                    self._log_update()

                self._validate()

                if not self._stopper.start_new_minibatch():
                    break

            if self._validation_iter is None:
                self._set_candidate_state()

            epoch_duration = time() - epoch_start_time
            epoch_minutes = epoch_duration / 60
            epoch_time_h, epoch_time_m = divmod(epoch_minutes, 60)
            message = "Finished training epoch {} in {:.0f} hours {:.1f} minutes." \
                      .format(self.epoch_number, epoch_time_h, epoch_time_m)
            best_cost = self.candidate_cost()
            if not best_cost is None:
                message += " Best validation perplexity {:.2f}.".format(
                    best_cost)
            print(message)

            self.epoch_number += 1
            self.update_number = 0

        duration = time() - start_time
        minutes = duration / 60
        time_h, time_m = divmod(minutes, 60)
        print("Training finished in {:.0f} hours {:.1f} minutes." \
              .format(time_h, time_m))

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

        if not self._network is None:
            self._network.get_state(state)
        self._training_iter.get_state(state)
        if not self._optimizer is None:
            self._optimizer.get_state(state)

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

        self._network.set_state(self._candidate_state)

        if not 'trainer' in self._candidate_state:
            raise IncompatibleStateError("Training state is missing.")
        h5_trainer = self._candidate_state['trainer']

        if not 'epoch_number' in h5_trainer.attrs:
            raise IncompatibleStateError("Current epoch number is missing from "
                                         "training state.")
        self.epoch_number = int(h5_trainer.attrs['epoch_number'])

        if not 'update_number' in h5_trainer.attrs:
            raise IncompatibleStateError("Current update number is missing "
                                         "from training state.")
        self.update_number = int(h5_trainer.attrs['update_number'])

        logging.info("[%d] (%.2f %%) of epoch %d",
                     self.update_number,
                     self.update_number / self._updates_per_epoch * 100,
                     self.epoch_number)

        if 'cost_history' in h5_trainer:
            self._cost_history = h5_trainer['cost_history'].value
            if self._cost_history.size == 0:
                print("Validation set cost history is empty in the training state.")
                self._candidate_index = None
            else:
                self._candidate_index = self._cost_history.size - 1
                self._log_validation()
        else:
            print("Warning: Validation set cost history is missing from "
                  "training state. Initializing to empty cost history.")
            self._cost_history = numpy.asarray([], dtype=theano.config.floatX)
            self._candidate_index = None

        self._training_iter.set_state(self._candidate_state)
        self._optimizer.set_state(self._candidate_state)

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
        old_value = self._optimizer.learning_rate
        new_value = old_value / 2
        self._reset_state()
        self._stopper.improvement_ceased()
        self._optimizer.learning_rate = new_value

        print("Model performance stopped improving. Decreasing learning rate "
              "from {} to {} and resetting state to {:.0f} % of epoch {}."
              .format(old_value,
                      new_value,
                      self.update_number / self._updates_per_epoch * 100,
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
                     self.update_number / self._updates_per_epoch * 100,
                     self.epoch_number,
                     self._optimizer.learning_rate,
                     self._optimizer.update_cost,
                     self._update_duration * 100)

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

        if self._cost_history.size == 0:
            self._candidate_index = None
        else:
            self._candidate_index = self._cost_history.size - 1

        self._candidate_state.flush()
        logging.info("New candidate for optimal state saved to %s.",
                     self._candidate_state.filename)

    def _validate(self):
        """If at or just before the actual validation point, computes perplexity
        and adds to the list of samples. At the actual validation point we have
        `self._samples_per_validation` values and combine them using
        `self._statistic_function`. If the model performance has improved, the
        state at the center of the validation samples will be saved using
        `self._set_candidate_state()`.

        :type perplexity: float
        :param perplexity: computed perplexity at a validation point, None
                           elsewhere
        """

        if self._validation_iter is None:
            return  # Validation has not been configured.

        if not self._is_scheduled(self._options['validation_frequency'],
                                  self._samples_per_validation - 1):
            return  # We don't have to validate now.

        perplexity = self._scorer.compute_perplexity(self._validation_iter)
        if numpy.isnan(perplexity) or numpy.isinf(perplexity):
            raise NumberError("Validation set perplexity computation resulted "
                              "in a numerical error.")

        self._local_perplexities.append(perplexity)
        if len(self._local_perplexities) == 1:
            logging.debug("[%d] First validation sample, perplexity %.2f.",
                          self.update_number,
                          perplexity)

        # The rest of the function will be executed only at and after the center
        # of sampling points.
        if not self._is_scheduled(self._options['validation_frequency'],
                                  self._samples_per_validation // 2):
            return

        # The first sampling point within samples_per_validation / 2 of the
        # actual validation point is the center of the sampling points. This
        # will be saved in case the model performance has improved.
        if self._validation_state is None:
            logging.debug("[%d] Center of validation, perplexity %.2f.",
                          self.update_number,
                          perplexity)
            self._validation_state = h5py.File(
                name='validation-state', driver='core', backing_store=False)
            self.get_state(self._validation_state)

        # The rest of the function will be executed only at the final sampling
        # point.
        if not self._is_scheduled(self._options['validation_frequency']):
            return
        logging.debug("[%d] Last validation sample, perplexity %.2f.",
                      self.update_number,
                      perplexity)

        if len(self._local_perplexities) < self._samples_per_validation:
            # After restoring a previous validation state, which is at the
            # center of the sampling points, the trainer will collect again half
            # of the samples. Don't take that as a validation.
            logging.debug("[%d] Only %d samples collected. Ignoring this "
                          "validation.",
                          self.update_number,
                          len(self._local_perplexities))
            self._local_perplexities = []
            self._validation_state.close()
            self._validation_state = None
            return

        statistic = self._statistic_function(self._local_perplexities)
        self._cost_history = numpy.append(self._cost_history, statistic)
        if self._has_improved():
            # Take the state at the actual validation point and replace the cost
            # history with the current cost history that also includes this
            # latest statistic.
            h5_cost_history = self._validation_state['trainer/cost_history']
            h5_cost_history.resize(self._cost_history.shape)
            h5_cost_history[:] = self._cost_history
            self._set_candidate_state(self._validation_state)

        self._log_validation()

        if (self._options['patience'] >= 0) and \
           (self.validations_since_candidate() > self._options['patience']):
            # Too many validations without finding a new candidate state.

            # If any validations have been done, the best state has been found
            # and saved. If training has been started from previous state,
            # _candidate_state has been set to the initial state.
            assert not self._candidate_state is None

            self._decrease_learning_rate()

        self._local_perplexities = []
        self._validation_state.close()
        self._validation_state = None

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

        modulo = self.update_number * frequency % self._updates_per_epoch
        return modulo < frequency or \
               self._updates_per_epoch - modulo <= within * frequency
