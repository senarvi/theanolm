#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import subprocess
import logging
import numpy
import theano
import theanolm
import theanolm.trainers as trainers
from filetypes import TextFileType

class TrainingProcess(object):
    def __init__(self, dictionary, network, scorer, validation_iter,
                 initial_state, args):
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

        :type args: dict
        :param args: a dictionary of command line arguments
        """

        self.network = network
        self.scorer = scorer
        self.validation_iter = validation_iter

        training_options = {
            'epsilon': args.numerical_stability_term,
            'gradient_decay_rate': args.gradient_decay_rate,
            'sqr_gradient_decay_rate': args.sqr_gradient_decay_rate,
            'learning_rate': args.learning_rate,
            'momentum': args.momentum}
        if not args.gradient_normalization is None:
            training_options['max_gradient_norm'] = args.gradient_normalization

        if args.optimization_method == 'sgd':
            self.trainer = trainers.SGDTrainer(self.network,
                                               training_options,
                                               args.profile)
        elif args.optimization_method == 'nesterov':
            self.trainer = trainers.NesterovTrainer(self.network,
                                                    training_options,
                                                    args.profile)
        elif args.optimization_method == 'adadelta':
            self.trainer = trainers.AdadeltaTrainer(self.network,
                                                    training_options,
                                                    args.profile)
        elif args.optimization_method == 'rmsprop-sgd':
            self.trainer = trainers.RMSPropSGDTrainer(self.network,
                                                      training_options,
                                                      args.profile)
        elif args.optimization_method == 'rmsprop-momentum':
            self.trainer = trainers.RMSPropMomentumTrainer(self.network,
                                                           training_options,
                                                           args.profile)
        elif args.optimization_method == 'adam':
            self.trainer = trainers.AdamTrainer(self.network,
                                                training_options,
                                                args.profile)
        else:
            raise ValueError("Invalid optimization method requested: " + \
                args.optimization_method)

        if not initial_state is None:
            print("Restoring training to previous state.")
            sys.stdout.flush()
            self.trainer.set_state(initial_state)

        print("Finding sentence start positions in training data.")
        sys.stdout.flush()
        sentence_starts = theanolm.find_sentence_starts(args.training_file)

        self.training_iter = theanolm.ShufflingBatchIterator(
            args.training_file,
            dictionary,
            sentence_starts,
            batch_size=args.batch_size,
            max_sequence_length=args.sequence_length)

        self.state_path = args.state_path
        self.model_path = args.model_path

        self.network_state_min_cost = None
        self.trainer_state_min_cost = None

        self.log_update_interval = args.log_update_interval
        self.save_interval = args.save_interval

        if args.validation_interval.endswith('e'):
            self.validation_interval_updates = 0
            self.validation_interval_epochs = int(args.validation_interval[:-1])
            print("Performing validation every %d epochs." % \
                  self.validation_interval_epochs)
        else:
            self.validation_interval_updates = int(args.validation_interval)
            self.validation_interval_epochs = 0
            print("Performing validation every %d updates." % \
                  self.validation_interval_updates)

        self.max_epochs = args.max_epochs
        self.wait_improvement = args.wait_improvement
        self.recall_when_annealing = args.recall_when_annealing
        self.reset_when_annealing = args.recall_when_annealing

    def save_training_state(self):
        """Saves current neural network and training state to disk.
        """

        path = self.state_path
        state = self.network.get_state()
        state.update(self.trainer.get_state())
        numpy.savez(path, **state)
        logging.info("Saved %d parameters to %s.", len(state), path)

    def save_model(self):
        """Saves the model parameters found to provide the minimum validatation
        set perplexity to disk.
        """

        if not self.network_state_min_cost is None:
            path = self.model_path
            state = self.network_state_min_cost
            numpy.savez(path, **state)
            logging.info("Saved %d parameters to %s.", len(state), path)

    def run(self):
        local_perplexities = None

        while self.trainer.epoch_number <= self.max_epochs:
            self.epoch_start_ppl = \
                self.scorer.compute_perplexity(self.validation_iter)
            print("Validation set perplexity at the start of epoch {0}/{1}: {2}"
                  "".format(self.trainer.epoch_number,
                            self.max_epochs,
                            self.epoch_start_ppl))

            while self.trainer.update_minibatch(self.training_iter):
                if (self.log_update_interval >= 1) and \
                   (self.trainer.total_updates % self.log_update_interval == 0):
                    self.trainer.log_update()

                if (self.validation_interval_updates >= 1) and \
                   (self.trainer.total_updates % self.validation_interval_updates == 0):
                    local_perplexities = []

                if not local_perplexities is None:
                    ppl = self.scorer.compute_perplexity(self.validation_iter)
                    local_perplexities.append(ppl)
                    if len(local_perplexities) >= 10:
                        ppl = numpy.mean(numpy.asarray(local_perplexities))
                        self._validate(ppl)
                        local_perplexities = None

                if (self.save_interval >= 1) and \
                   (self.trainer.total_updates % self.save_interval == 0):
                    # Save the best parameters and the current state.
                    self.save_model()
                    self.save_training_state()

            if (self.validation_interval_epochs >= 1) and \
               (self.trainer.epoch_number % self.validation_interval_epochs == 0):
                self._validate()

        print("Stopping because %d epochs was reached." % self.trainer.epoch_number)
        validation_ppl = self.scorer.compute_perplexity(self.validation_iter)
        self.trainer.append_validation_cost(validation_ppl)
        if self.trainer.validations_since_min_cost() == 0:
            self.network_state_min_cost = self.network.get_state()

    def _validate(self, validation_ppl):
        if numpy.isnan(validation_ppl) or numpy.isinf(validation_ppl):
            raise NumberError("Validation set perplexity computation resulted "
                              "in a numerical error.")

        self.trainer.append_validation_cost(validation_ppl)
        validations_since_best = self.trainer.validations_since_min_cost()
        if validations_since_best == 0:
            # This is the minimum cost so far.
            self.network_state_min_cost = self.network.get_state()
            self.trainer_state_min_cost = self.trainer.get_state()
        elif (self.wait_improvement >= 0) and \
             (validations_since_best > self.wait_improvement):
            # Too many validations without improvement.
            if self.recall_when_annealing:
                self.network.set_state(self.network_state_min_cost)
                self.trainer.set_state(self.trainer_state_min_cost)
# XXX       if validation_ppl >= initial_ppl:
# XXX           self.trainer.reset()
# XXX           self.trainer.reset_cost_history()
# XXX       else:
# XXX           self.trainer.decrease_learning_rate()
# XXX           self.trainer.reset()
# XXX           self.trainer.reset_cost_history()
            self.trainer.decrease_learning_rate()
            if self.reset_when_annealing:
                self.trainer.reset()


parser = argparse.ArgumentParser()

argument_group = parser.add_argument_group("files")
argument_group.add_argument(
    'model_path', metavar='MODEL', type=str,
    help='path where the best model state will be saved in numpy .npz format')
argument_group.add_argument(
    'training_file', metavar='TRAINING-SET', type=TextFileType('r'),
    help='text or .gz file containing training data (one sentence per line)')
argument_group.add_argument(
    'validation_file', metavar='VALIDATION-SET', type=TextFileType('r'),
    help='text or .gz file containing validation data (one sentence per line) '
         'for early stopping')
argument_group.add_argument(
    'dictionary_file', metavar='DICTIONARY', type=TextFileType('r'),
    help='text or .gz file containing word list (one word per line) or word to '
         'class ID mappings (word and ID per line)')
argument_group.add_argument(
    '--training-state', dest='state_path', metavar='FILE', type=str, default=None,
    help='the last training state will be read from and written to FILE in '
         'numpy .npz format (if not given, starts from scratch and only saves '
         'the best model)')
argument_group.add_argument(
    '--dictionary-format', metavar='FORMAT', type=str, default='words',
    help='dictionary format, one of "words" (one word per line, default), '
         '"classes" (word and class ID per line), "srilm-classes" (class '
         'name, membership probability, and word per line)')

argument_group = parser.add_argument_group("network structure")
argument_group.add_argument(
    '--word-projection-dim', metavar='N', type=int, default=100,
    help='word projections will be N-dimensional (default 100)')
argument_group.add_argument(
    '--hidden-layer-size', metavar='N', type=int, default=1000,
    help='hidden layer will contain N outputs (default 1000)')
argument_group.add_argument(
    '--hidden-layer-type', metavar='TYPE', type=str, default='lstm',
    help='hidden layer unit type, "lstm" or "gru" (default "lstm")')
argument_group.add_argument(
    '--skip-layer-size', metavar='N', type=int, default=0,
    help='if N is greater than zero, include a layer between hidden and output '
         'layers, with N outputs, and direct connections from input layer '
         '(default is to not create such layer)')

argument_group = parser.add_argument_group("training")
argument_group.add_argument(
    '--sequence-length', metavar='N', type=int, default=100,
    help='ignore sentences longer than N words (default 100)')
argument_group.add_argument(
    '--batch-size', metavar='N', type=int, default=16,
    help='each mini-batch will contain N sentences (default 16)')
argument_group.add_argument(
    '--validation-interval', metavar='N', type=str, default='1e',
    help='cross-validation for reducing learning rate is performed after every '
         'Nth mini-batch update, or every Nth epoch if N is followed by the '
         'unit "e" (default is "1e", meaning after every epoch)')
argument_group.add_argument(
    '--wait-improvement', metavar='N', type=int, default=0,
    help='wait for N validations, before decreasing learning rate, if '
         'perplexity has not decreased; if less than zero, never decrease '
         'learning rate (default is 0, meaning that learning rate will be '
         'decreased immediately when perplexity stops decreasing)')
argument_group.add_argument(
    '--recall-when-annealing', action="store_true",
    help='restore the state of minimum validation cost when decreasing '
         'learning rate (default is to continue with the current state, which '
         'is better if learning rate is reduced hastily)')
argument_group.add_argument(
    '--reset-when-annealing', action="store_true",
    help='reset the optimizer timestep when decreasing learning rate')
argument_group.add_argument(
    '--max-epochs', metavar='N', type=int, default=1000,
    help='perform at most N training epochs (default 1000)')
argument_group.add_argument(
    '--optimization-method', metavar='METHOD', type=str, default='adam',
    help='optimization method, one of "sgd", "nesterov", "adadelta", '
         '"rmsprop-sgd", "rmsprop-momentum", "adam" (default "adam")')
argument_group.add_argument(
    '--learning-rate', metavar='ALPHA', type=float, default=0.001,
    help='initial learning rate (default 0.001)')
argument_group.add_argument(
    '--momentum', metavar='BETA', type=float, default=0.9,
    help='momentum coefficient for momentum optimization methods (default 0.9)')
argument_group.add_argument(
    '--gradient-decay-rate', metavar='GAMMA', type=float, default=0.9,
    help='geometric rate for averaging gradients (default 0.9)')
argument_group.add_argument(
    '--sqr-gradient-decay-rate', metavar='GAMMA', type=float, default=0.999,
    help='geometric rate for averaging squared gradients in Adam optimizer '
         '(default 0.999)')
argument_group.add_argument(
    '--numerical-stability-term', metavar='EPSILON', type=float, default=1e-7,
    help='a value that is used to prevent instability when dividing by very '
         'small numbers (default 1e-7)')
argument_group.add_argument(
    '--gradient-normalization', metavar='THRESHOLD', type=float, default=None,
    help='scale down the gradients if necessary to make sure their norm '
         '(normalized by mini-batch size) will not exceed THRESHOLD (no '
         'scaling by default)')
argument_group.add_argument(
    '--save-interval', metavar='N', type=int, default=1000,
    help='save training state after every Nth mini-batch update; if less than '
         'one, save the model only after training (default 1000)')
argument_group.add_argument(
    '--random-seed', metavar='N', type=int, default=None,
    help='seed to initialize the random state (default is to seed from a '
         'random source provided by the oprating system)')

argument_group = parser.add_argument_group("logging and debugging")
argument_group.add_argument(
    '--log-file', metavar='FILE', type=str, default='-',
    help='path where to write log file (default is standard output)')
argument_group.add_argument(
    '--log-level', metavar='LEVEL', type=str, default='info',
    help='minimum level of events to log, one of "debug", "info", "warn" '
         '(default "info")')
argument_group.add_argument(
    '--log-update-interval', metavar='N', type=int, default=1000,
    help='print statistics of every Nth mini-batch update; quiet if less than '
         'one (default 1000)')
argument_group.add_argument(
    '--debug', action="store_true",
    help='enables debugging Theano errors')
argument_group.add_argument(
    '--profile', action="store_true",
    help='enables profiling Theano functions')

args = parser.parse_args()

numpy.random.seed(args.random_seed)

log_file = args.log_file
log_level = getattr(logging, args.log_level.upper(), None)
if not isinstance(log_level, int):
    raise ValueError("Invalid logging level requested: " + args.log_level)
log_format = '%(asctime)s %(funcName)s: %(message)s'
if args.log_file == '-':
    logging.basicConfig(stream=sys.stdout, format=log_format, level=log_level)
else:
    logging.basicConfig(filename=log_file, format=log_format, level=log_level)

if args.debug:
    theano.config.compute_test_value = 'warn'
else:
    theano.config.compute_test_value = 'off'

try:
    script_path = os.path.dirname(os.path.realpath(__file__))
    git_description = subprocess.check_output(['git', 'describe'], cwd=script_path)
    print("TheanoLM", git_description.decode('utf-8'))
except subprocess.CalledProcessError:
    logging.warn("Git repository description is not available.")

if (not args.state_path is None) and os.path.exists(args.state_path):
    print("Reading previous state from %s." % args.state_path)
    initial_state = numpy.load(args.state_path)
else:
    initial_state = None

print("Reading dictionary.")
sys.stdout.flush()
dictionary = theanolm.Dictionary(args.dictionary_file, args.dictionary_format)
print("Number of words in vocabulary:", dictionary.num_words())
print("Number of word classes:", dictionary.num_classes())

print("Building neural network.")
sys.stdout.flush()
architecture = theanolm.Network.Architecture(
    args.word_projection_dim,
    args.hidden_layer_type,
    args.hidden_layer_size,
    args.skip_layer_size)
print(architecture)
network = theanolm.Network(dictionary, architecture, args.profile)
if not initial_state is None:
    print("Restoring neural network to previous state.")
    sys.stdout.flush()
    network.set_state(initial_state)

print("Building text scorer.")
sys.stdout.flush()
scorer = theanolm.TextScorer(network, args.profile)

validation_iter = theanolm.BatchIterator(args.validation_file, dictionary)

print("Building neural network trainer.")
sys.stdout.flush()
process = TrainingProcess(dictionary, network, scorer, validation_iter, initial_state, args)

print("Training neural network.")
sys.stdout.flush()
process.run()

print("Saving neural network and training state.")
sys.stdout.flush()
process.save_training_state()
if process.network_state_min_cost is None:
    print("Validation set perplexity did not decrease during training.")
else:
    process.save_model()
    network.set_state(process.network_state_min_cost)
    validation_ppl = scorer.compute_perplexity(validation_iter)
    print("Best validation set perplexity:", validation_ppl)
