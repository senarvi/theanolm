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
from theanolm.trainers import create_trainer
from filetypes import TextFileType


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
    '--hidden-layer-type', metavar='NAME', type=str, default='lstm',
    help='hidden layer unit type, "lstm" or "gru" (default "lstm")')
argument_group.add_argument(
    '--skip-layer-size', metavar='N', type=int, default=0,
    help='if N is greater than zero, include a layer between hidden and output '
         'layers, with N outputs, and direct connections from input layer '
         '(default is to not create such layer)')

argument_group = parser.add_argument_group("training")
argument_group.add_argument(
    '--training-strategy', metavar='NAME', type=str, default='basic',
    help='selects a training and validation strategy, one of "basic", '
        '"local-mean", "local-median", "validation-average" (default "basic")')
argument_group.add_argument(
    '--stopping-criterion', metavar='NAME', type=str, default='basic',
    help='selects a criterion for early-stopping, one of "basic" (fixed number '
         'of epochs), "learning-rate", "patience" (default "basic")')
argument_group.add_argument(
    '--sequence-length', metavar='N', type=int, default=100,
    help='ignore sentences longer than N words (default 100)')
argument_group.add_argument(
    '--batch-size', metavar='N', type=int, default=16,
    help='each mini-batch will contain N sentences (default 16)')
argument_group.add_argument(
    '--validation-frequency', metavar='N', type=int, default='100',
    help='cross-validate for reducing learning rate N times per training epoch '
         '(default 100)')
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
    '--random-seed', metavar='N', type=int, default=None,
    help='seed to initialize the random state (default is to seed from a '
         'random source provided by the oprating system)')

argument_group = parser.add_argument_group("optimization")
argument_group.add_argument(
    '--optimization-method', metavar='NAME', type=str, default='adam',
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

if os.path.exists(args.model_path):
    print("Reading initial state from %s." % args.model_path)
    initial_state = numpy.load(args.model_path)
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

optimization_options = {
    'method': args.optimization_method,
    'epsilon': args.numerical_stability_term,
    'gradient_decay_rate': args.gradient_decay_rate,
    'sqr_gradient_decay_rate': args.sqr_gradient_decay_rate,
    'learning_rate': args.learning_rate,
    'momentum': args.momentum}
if not args.gradient_normalization is None:
    optimization_options['max_gradient_norm'] = args.gradient_normalization

training_options = {
    'strategy': args.training_strategy,
    'stopping_criterion': args.stopping_criterion,
    'batch_size': args.batch_size,
    'sequence_length': args.sequence_length,
    'validation_frequency': args.validation_frequency,
    'wait_improvement': args.wait_improvement,
    'recall_when_annealing': args.recall_when_annealing,
    'reset_when_annealing': args.recall_when_annealing,
    'max_epochs': args.max_epochs}

print("Building neural network trainer.")
sys.stdout.flush()
trainer = create_trainer(training_options, optimization_options,
    network, dictionary, scorer,
    args.training_file, validation_iter,
    args.profile)
if not initial_state is None:
    print("Restoring training to previous state.")
    sys.stdout.flush()
    trainer.set_state(initial_state)
trainer.set_model_path(args.model_path)
trainer.set_logging(args.log_update_interval)

print("Training neural network.")
sys.stdout.flush()
trainer.run()

if trainer.min_cost_state is None:
    print("Validation set perplexity did not decrease during training.")
else:
    network.set_state(trainer.min_cost_state)
    validation_ppl = scorer.compute_perplexity(validation_iter)
    print("Best validation set perplexity:", validation_ppl)
