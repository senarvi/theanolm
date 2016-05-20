#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import mmap
import logging
import numpy
import h5py
import theano
from theanolm import Vocabulary, Architecture, Network, TextScorer
from theanolm import LinearBatchIterator
from theanolm.trainers import create_trainer
from theanolm.filetypes import TextFileType

def add_arguments(parser):
    argument_group = parser.add_argument_group("files")
    argument_group.add_argument(
        'model_path', metavar='MODEL-FILE', type=str,
        help='path where the best model state will be saved in numpy .npz '
             'format')
    argument_group.add_argument(
        'validation_file', metavar='VALID-FILE', type=TextFileType('r'),
        help='text or .gz file containing validation data (one sentence per '
             'line) for early stopping')
    argument_group.add_argument(
        '--training-set', metavar='FILE', type=TextFileType('r'), nargs='+',
        required=True,
        help='text or .gz files containing training data (one sentence per '
             'line)')
    argument_group.add_argument(
        '--vocabulary', metavar='FILE', type=str, default=None,
        help='text or .gz file containing word list or class definitions '
             '(default is to include all the files from the training set)')
    argument_group.add_argument(
        '--vocabulary-format', metavar='FORMAT', type=str, default='words',
        help='format of the file specified with --vocabulary argument, one of '
             '"words" (one word per line, default), "classes" (word and class '
             'ID per line), "srilm-classes" (class name, membership '
             'probability, and word per line)')

    argument_group = parser.add_argument_group("network architecture")
    argument_group.add_argument(
        '--architecture', metavar='FILE', type=str, default='lstm300',
        help='path to neural network architecture description, or a standard '
             'architecture name, "lstm300" or "lstm1500" (default "lstm300")')
    argument_group.add_argument(
        '--num-classes', metavar='N', type=int, default=None,
        help='generate N classes using a simple word frequency based algorithm '
             'when --vocabulary argument is not given (default is to not use '
             'word classes)')

    argument_group = parser.add_argument_group("training process")
    argument_group.add_argument(
        '--sampling', metavar='FRACTION', type=float, nargs='*', default=[],
        help='randomly sample only FRACTION of each training file on each '
             'epoch (list the fractions in the same order as the training '
             'files)')
    argument_group.add_argument(
        '--training-strategy', metavar='NAME', type=str, default='local-mean',
        help='selects a training and validation strategy, one of "basic", '
            '"local-mean", "local-median", "validation-average" (default '
            '"local-mean")')
    argument_group.add_argument(
        '--sequence-length', metavar='N', type=int, default=100,
        help='ignore sentences longer than N words (default 100)')
    argument_group.add_argument(
        '--batch-size', metavar='N', type=int, default=16,
        help='each mini-batch will contain N sentences (default 16)')
    argument_group.add_argument(
        '--validation-frequency', metavar='N', type=int, default='5',
        help='cross-validate for reducing learning rate or early stopping N '
             'times per training epoch (default 5)')
    argument_group.add_argument(
        '--patience', metavar='N', type=int, default=4,
        help='allow perplexity to increase N consecutive cross-validations, '
             'before decreasing learning rate; if less than zero, never '
             'decrease learning rate (default 4)')
    argument_group.add_argument(
        '--random-seed', metavar='N', type=int, default=None,
        help='seed to initialize the random state (default is to seed from a '
             'random source provided by the oprating system)')
    
    argument_group = parser.add_argument_group("optimization")
    argument_group.add_argument(
        '--optimization-method', metavar='NAME', type=str, default='adagrad',
        help='optimization method, one of "sgd", "nesterov", "adagrad", '
             '"adadelta", "rmsprop-sgd", "rmsprop-nesterov", "adam" '
             '(default "adagrad")')
    argument_group.add_argument(
        '--learning-rate', metavar='ALPHA', type=float, default=0.1,
        help='initial learning rate (default 0.1)')
    argument_group.add_argument(
        '--momentum', metavar='BETA', type=float, default=0.9,
        help='momentum coefficient for momentum optimization methods (default '
             '0.9)')
    argument_group.add_argument(
        '--gradient-decay-rate', metavar='GAMMA', type=float, default=0.9,
        help='geometric rate for averaging gradients (default 0.9)')
    argument_group.add_argument(
        '--sqr-gradient-decay-rate', metavar='GAMMA', type=float, default=0.999,
        help='geometric rate for averaging squared gradients in Adam optimizer '
             '(default 0.999)')
    argument_group.add_argument(
        '--numerical-stability-term', metavar='EPSILON', type=float,
        default=1e-6,
        help='a value that is used to prevent instability when dividing by '
             'very small numbers (default 1e-6)')
    argument_group.add_argument(
        '--gradient-normalization', metavar='THRESHOLD', type=float,
        default=5,
        help='scale down the gradients if necessary to make sure their norm '
             '(normalized by mini-batch size) will not exceed THRESHOLD '
             '(default 5)')
    argument_group.add_argument(
        '--unk-penalty', metavar='LOGPROB', type=float, default=None,
        help="if LOGPROB is zero, do not include <unk> tokens in perplexity "
             "computation; otherwise use constant LOGPROB as <unk> token score "
             "(default is to use the network to predict <unk> probability)")
    argument_group.add_argument(
        '--weights', metavar='WEIGHT', type=float, nargs='*', default=[],
        help='scale a mini-batch update by WEIGHT if the data is from the '
             'corresponding training file (list the weights in the same order '
             'as the training files)')

    argument_group = parser.add_argument_group("early stopping")
    argument_group.add_argument(
        '--stopping-criterion', metavar='NAME', type=str,
        default='annealing-count',
        help='selects a criterion for early-stopping, one of "epoch-count" '
             '(fixed number of epochs), "no-improvement" (no improvement since '
             'learning rate was decreased), "annealing-count" (default, '
             'learning rate is decreased a fixed number of times)')
    argument_group.add_argument(
        '--min-epochs', metavar='N', type=int, default=1,
        help='perform at least N training epochs (default 1)')
    argument_group.add_argument(
        '--max-epochs', metavar='N', type=int, default=100,
        help='perform at most N training epochs (default 100)')
    argument_group.add_argument(
        '--max-annealing-count', metavar='N', type=int, default=0,
        help='when using annealing-count stopping criterion, continue training '
             'after decreasing learning rate at most N times (default 0)')
    
    argument_group = parser.add_argument_group("logging and debugging")
    argument_group.add_argument(
        '--log-file', metavar='FILE', type=str, default='-',
        help='path where to write log file (default is standard output)')
    argument_group.add_argument(
        '--log-level', metavar='LEVEL', type=str, default='info',
        help='minimum level of events to log, one of "debug", "info", "warn" '
             '(default "info")')
    argument_group.add_argument(
        '--log-interval', metavar='N', type=int, default=1000,
        help='print statistics of every Nth mini-batch update; quiet if less '
             'than one (default 1000)')
    argument_group.add_argument(
        '--debug', action="store_true",
        help='enables debugging Theano errors')
    argument_group.add_argument(
        '--profile', action="store_true",
        help='enables profiling Theano functions')

def train(args):
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
    theano.config.profile = args.profile
    theano.config.profile_memory = args.profile

    with h5py.File(args.model_path, 'a', driver='core') as state:
        if state.keys():
            print("Reading vocabulary from existing network state.")
            sys.stdout.flush()
            vocabulary = Vocabulary.from_state(state)
        elif args.vocabulary is None:
            print("Constructing vocabulary from training set.")
            sys.stdout.flush()
            vocabulary = Vocabulary.from_corpus(args.training_set,
                                                args.num_classes)
            for training_file in args.training_set:
                training_file.seek(0)
            vocabulary.get_state(state)
        else:
            print("Reading vocabulary from {}.".format(args.vocabulary))
            sys.stdout.flush()
            with open(args.vocabulary, 'rt', encoding='utf-8') as vocab_file:
                vocabulary = Vocabulary.from_file(vocab_file,
                                                  args.vocabulary_format)
                if args.vocabulary_format == 'classes':
                    print("Computing class membership probabilities from "
                          "unigram word counts.")
                    sys.stdout.flush()
                    vocabulary.compute_probs(args.training_set)
            vocabulary.get_state(state)
        print("Number of words in vocabulary:", vocabulary.num_words())
        print("Number of word classes:", vocabulary.num_classes())

        print("Building neural network.")
        sys.stdout.flush()
        if args.architecture == 'lstm300' or args.architecture == 'lstm1500':
            architecture = Architecture.from_package(args.architecture)
        else:
            with open(args.architecture, 'rt', encoding='utf-8') as arch_file:
                architecture = Architecture.from_description(arch_file)
        network = Network(vocabulary, architecture, batch_processing=True,
                          profile=args.profile)

        sys.stdout.flush()
        if args.unk_penalty is None:
            ignore_unk = False
            unk_penalty = None
        elif args.unk_penalty == 0:
            ignore_unk = True
            unk_penalty = None
        else:
            ignore_unk = False
            unk_penalty = args.unk_penalty

        num_training_files = len(args.training_set)
        if len(args.weights) > num_training_files:
            print("You specified more weights than training files.")
            sys.exit(1)
        weights = numpy.ones(num_training_files).astype(theano.config.floatX)
        for index, weight in enumerate(args.weights):
            weights[index] = weight

        print("Building text scorer.")
        scorer = TextScorer(network, ignore_unk, unk_penalty, args.profile)

        validation_mmap = mmap.mmap(args.validation_file.fileno(),
                                    0,
                                    prot=mmap.PROT_READ)
        validation_iter = LinearBatchIterator(validation_mmap,
                                              vocabulary,
                                              batch_size=32)

        optimization_options = {
            'method': args.optimization_method,
            'epsilon': args.numerical_stability_term,
            'gradient_decay_rate': args.gradient_decay_rate,
            'sqr_gradient_decay_rate': args.sqr_gradient_decay_rate,
            'learning_rate': args.learning_rate,
            'weights': weights,
            'momentum': args.momentum,
            'max_gradient_norm': args.gradient_normalization,
            'ignore_unk': ignore_unk,
            'unk_penalty': unk_penalty
        }
        logging.debug("OPTIMIZATION OPTIONS")
        for option_name, option_value in optimization_options.items():
            if type(option_value) is list:
                value_str = ', '.join(str(x) for x in option_value)
                logging.debug("%s: [%s]", option_name, value_str)
            else:
                logging.debug("%s: %s", option_name, str(option_value))

        training_options = {
            'strategy': args.training_strategy,
            'batch_size': args.batch_size,
            'sequence_length': args.sequence_length,
            'validation_frequency': args.validation_frequency,
            'patience': args.patience,
            'stopping_criterion': args.stopping_criterion,
            'max_epochs': args.max_epochs,
            'min_epochs': args.min_epochs,
            'max_annealing_count': args.max_annealing_count
        }
        logging.debug("TRAINING OPTIONS")
        for option_name, option_value in training_options.items():
            logging.debug("%s: %s", option_name, str(option_value))

        print("Building neural network trainer.")
        sys.stdout.flush()
        if len(args.sampling) > len(args.training_set):
            print("You specified more sampling coefficients than training "
                  "files.")
            sys.exit(1)
        trainer = create_trainer(
            training_options, optimization_options,
            network, vocabulary, scorer,
            args.training_set, args.sampling, validation_iter,
            state, args.profile)
        trainer.set_logging(args.log_interval)

        print("Training neural network.")
        sys.stdout.flush()
        trainer.run()

        if not state.keys():
            print("The model has not been trained.")
        else:
            network.set_state(state)
            perplexity = scorer.compute_perplexity(validation_iter)
            print("Best validation set perplexity:", perplexity)
