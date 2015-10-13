#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import sys
import logging
import numpy
from theanolm import find_sentence_starts, ShufflingBatchIterator
from theanolm.exceptions import IncompatibleStateError, NumberError
from theanolm.optimizers import create_optimizer
from theanolm.trainers.stoppingcriteria import create_stopper

class BasicTrainer(object):
    """Basic training process saves a history of validation costs and "
    decreases learning rate when the cost does not decrease anymore.
    """

    def __init__(self, training_options, optimization_options,
                 network, dictionary, scorer,
                 training_file, validation_iter,
                 initial_state,
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

        :type initial_state: dict
        :param initial_state: if not None, the trainer will be initialized with
                              the parameters loaded from this dictionary

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
        if not initial_state is None:
            print("Restoring training to previous state.")
            sys.stdout.flush()
            self.optimizer.set_state(initial_state)

        print("Finding sentence start positions in training data.")
        sys.stdout.flush()
        sentence_starts = find_sentence_starts(training_file)

        self.training_iter = ShufflingBatchIterator(
            training_file,
            dictionary,
            sentence_starts,
            batch_size=training_options['batch_size'],
            max_sequence_length=training_options['sequence_length'])

        print("Computing the number of training updates per epoch.")
        sys.stdout.flush()
        self.updates_per_epoch = len(self.training_iter)

        self.stopper = create_stopper(training_options, self)

        self.options = training_options

        self.state_path = None
        self.save_frequency = 0
        self.model_path = None
        self.log_update_interval = 0

        self.network_state_min_cost = None
        self.optimizer_state_min_cost = None

        # current training epoch
        self.epoch_number = 1
        # number of mini-batch updates performed in this epoch
        self.update_number = 0
        # total number of mini-batch updates performed (after restart)
        self.total_updates = 0
        # validation set cost history
        self._cost_history = []

    def set_state_saving(self, path, frequency):
        self.state_path = path
        self.save_frequency = frequency

    def set_model_saving(self, path):
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

                if self._is_scheduled(self.save_frequency):
                    self.save_training_state()

                if not self.stopper.start_new_minibatch():
                    break

            self.epoch_number += 1
            self.update_number = 0

        logging.info("Training finished.")
        validation_ppl = self.scorer.compute_perplexity(self.validation_iter)
        self._append_validation_cost(validation_ppl)
        if self._validations_since_min_cost() == 0:
            self.network_state_min_cost = self.network.get_state()
            self.save_model()

        self.save_training_state()

    def log_update(self):
        """Logs information about the previous mini-batch update.
        """

        logging.info("Update %d (%.2f %%) of epoch %d -- "
                     "lr = %g, cost = %.2f, duration = %.2f ms",
                     self.update_number,
                     self.update_number / self.updates_per_epoch * 100,
                     self.epoch_number,
                     self.optimizer.get_learning_rate(),
                     self.optimizer.update_cost,
                     self.optimizer.update_duration * 100)

    def save_training_state(self):
        """Saves current neural network and training state to disk.
        """

        path = self.state_path
        if not (path is None):
            state = self.network.get_state()
            state.update(self.get_state())
            numpy.savez(path, **state)
            logging.info("Saved %d parameters to %s.", len(state), path)

    def save_model(self):
        """Saves the model parameters, found to provide the minimum validatation
        set perplexity, to disk.
        """

        path = self.model_path
        state = self.network_state_min_cost
        if not (path is None or state is None):
            numpy.savez(path, **state)
            logging.info("Saved %d parameters to %s.", len(state), path)

    def get_state(self):
        """Pulls parameter values from Theano shared variables.

        For consistency, all the parameter values are returned as numpy types,
        since state read from a model file also contains numpy types.

        :rtype: dict of numpy types
        :returns: a dictionary of the parameter values
        """

        result = OrderedDict()
        result['epoch_number'] = numpy.int64(self.epoch_number)
        result['update_number'] = numpy.int64(self.update_number)
        result['cost_history'] = numpy.asarray(self._cost_history)
        result.update(self.optimizer.get_state())
        return result

    def set_state(self, state):
        """Sets the values of Theano shared variables.
        
        Requires that ``state`` contains values for all the training parameters.

        :type state: dict of numpy types
        :param state: a dictionary of training parameters
        """

        if not 'epoch_number' in state:
            raise IncompatibleStateError("Current epoch number is missing from "
                                         "training state.")
        self.epoch_number = state['epoch_number'].item()

        if not 'update_number' in state:
            raise IncompatibleStateError("Current update number is missing "
                                         "from training state.")
        self.update_number = state['update_number'].item()
        logging.info("Restored training state from update %d.%d.",
            self.epoch_number, self.update_number)

        if not 'cost_history' in state:
            raise IncompatibleStateError("Validation set cost history is "
                                         "missing from training state.")
        saved_cost_history = state['cost_history'].tolist()
        # If the error history was empty when the state was saved,
        # ndarray.tolist() will return None.
        if saved_cost_history is None:
            self._cost_history = []
        else:
            self._cost_history = saved_cost_history
        logging.debug("Validation set cost history since learning rate was "
                      "decreased:")
        logging.debug(str(numpy.asarray(self._cost_history)))

        self.optimizer.set_state(state)

    def decrease_learning_rate(self):
        """Called when the validation set cost stops decreasing.
        """

        self.optimizer.decrease_learning_rate()
        self._cost_history = []
        self.stopper.learning_rate_decreased()

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

    def _validate(self, perplexity):
        if perplexity is None:
            return

        self._append_validation_cost(perplexity)

        validations_since_best = self._validations_since_min_cost()
        if validations_since_best == 0:
            # This is the minimum cost so far.
            self.network_state_min_cost = self.network.get_state()
            self.trainer_state_min_cost = self.get_state()
            self.save_model()
        elif (self.options['wait_improvement'] >= 0) and \
             (validations_since_best > self.options['wait_improvement']):
            # Too many validations without improvement.
            if self.options['recall_when_annealing']:
                self.network.set_state(self.network_state_min_cost)
                self.set_state(self.trainer_state_min_cost)
# XXX       if cost >= self.epoch_start_ppl:
# XXX           self.optimizer.reset()
# XXX           self._cost_history = []
# XXX       else:
# XXX           self.decrease_learning_rate()
# XXX           self.optimizer.reset()
# XXX           self._cost_history = []
            self.decrease_learning_rate()
            if self.options['reset_when_annealing']:
                self.optimizer.reset()

    def _append_validation_cost(self, validation_cost):
        """Adds the validation set cost to the cost history.

        :type validation_cost: float
        :param validation_cost: the new validation set cost to be added to the history
        """

        self._cost_history.append(validation_cost)
        logging.debug("Validation set cost history since learning rate was "
                      "decreased:")
        logging.debug(str(numpy.asarray(self._cost_history)))

    def _validations_since_min_cost(self):
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

