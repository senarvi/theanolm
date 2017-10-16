#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the "theanolm train" command.
"""

import sys
import mmap
import logging

import h5py
import numpy
import theano

from theanolm import Vocabulary, Architecture, Network
from theanolm.backend import TextFileType
from theanolm.parsing import LinearBatchIterator
from theanolm.training import Trainer, create_optimizer, CrossEntropyCost, \
                              NCECost, BlackoutCost
from theanolm.scoring import TextScorer
from theanolm.vocabulary import compute_word_counts

def add_arguments(parser):
    """Specifies the command line arguments supported by the "theanolm train"
    command.

    :type parser: argparse.ArgumentParser
    :param parser: a command line argument parser
    """

    argument_group = parser.add_argument_group("data")
    argument_group.add_argument(
        'model_path', metavar='MODEL-FILE', type=str,
        help='path where the best model state will be saved in HDF5 binary '
             'data format')
    argument_group.add_argument(
        '--training-set', metavar='FILE', type=TextFileType('r'), nargs='+',
        required=True,
        help='text files containing training data (UTF-8, one sentence per '
             'line, assumed to be compressed if the name ends in ".gz")')
    argument_group.add_argument(
        '--validation-file', metavar='VALID-FILE', type=TextFileType('r'),
        default=None,
        help='text file containing validation data for early stopping (UTF-8, '
             'one sentence per line, assumed to be compressed if the name ends '
             'in ".gz")')

    argument_group = parser.add_argument_group("vocabulary")
    argument_group.add_argument(
        '--vocabulary', metavar='FILE', type=str, default=None,
        help='word or class vocabulary to be used in the neural network input '
             'and output, in the format specified by the --vocabulary-format '
             'argument (UTF-8 text, default is to use all the words from the '
             'training data)')
    argument_group.add_argument(
        '--vocabulary-format', metavar='FORMAT', type=str, default='words',
        help='format of the file specified with --vocabulary argument, one of '
             '"words" (one word per line, default), "classes" (word and class '
             'ID per line), "srilm-classes" (class name, membership '
             'probability, and word per line)')
    argument_group.add_argument(
        '--num-classes', metavar='N', type=int, default=None,
        help='generate N classes using a simple word frequency based algorithm '
             'when --vocabulary argument is not given (default is to not use '
             'word classes)')

    argument_group = parser.add_argument_group("network architecture")
    argument_group.add_argument(
        '--architecture', metavar='FILE', type=str, default='lstm300',
        help='path to neural network architecture description, or a standard '
             'architecture name, "lstm300" or "lstm1500" (default "lstm300")')

    argument_group = parser.add_argument_group("training process")
    argument_group.add_argument(
        '--sampling', metavar='FRACTION', type=float, nargs='*', default=[],
        help='randomly sample only FRACTION of each training file on each '
             'epoch (list the fractions in the same order as the training '
             'files)')
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
        '--cost', metavar='NAME', type=str, default='cross-entropy',
        help='cost function, one of "cross-entropy" (default), "nce" '
             '(noise-contrastive estimation), or "blackout"')
    argument_group.add_argument(
        '--num-noise-samples', metavar='K', type=int, default=5,
        help='sampling based costs sample K noise words per one training word '
             '(default 5)')
    argument_group.add_argument(
        '--noise-distribution', metavar='DIST', type=str, default='uniform',
        help='sample noise from DIST; one of "uniform" (default, but less '
             'accurate), "log-uniform" (the vocabulary should be ordered by '
             'decreasing frequency), "unigram" (unigram distribution of words '
             'in training data, slow)')
    argument_group.add_argument(
        '--noise-dampening', metavar='ALPHA', type=float, default=0.5,
        help='the empirical unigram distribution is raised to the power ALPHA '
             'before sampling noise words; 0.0 corresponds to the uniform '
             'distribution and 1.0 corresponds to the unigram distribution '
             '(only applicable with --noise-distribution=unigram, default 0.5)')
    argument_group.add_argument(
        '--noise-sharing', metavar='SHARING', type=str, default=None,
        help='can be "seq" for sharing noise samples between mini-batch '
             'sequences, or "batch" for sharing noise samples across einter '
             'mini-batch for improved speed (default is no sharing, which is '
             'very slow)')
    argument_group.add_argument(
        '--exclude-unk', action="store_true",
        help="exclude <unk> tokens from cost and perplexity computations")
    argument_group.add_argument(
        '--weights', metavar='LAMBDA', type=float, nargs='*', default=[],
        help='scale a mini-batch update by LAMBDA if the data is from the '
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

    argument_group = parser.add_argument_group("configuration")
    argument_group.add_argument(
        '--default-device', metavar='DEVICE', type=str, default=None,
        help='when multiple GPUs are present, use DEVICE as default')

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
        help='use test values to get better error messages from Theano')
    argument_group.add_argument(
        '--print-graph', action="store_true",
        help='print Theano computation graph')
    argument_group.add_argument(
        '--profile', action="store_true",
        help='enable profiling Theano functions')

def _read_vocabulary(args, state):
    """If ``state`` contains data, reads the vocabulary from the HDF5 state.
    Otherwise reads a vocabulary file or constructs the vocabulary from the
    training set and writes it to the HDF5 state.

    If the state does not contain data and --vocabulary argument is given, reads
    the vocabulary from the file given after the argument. The rest of the words
    in the training set will be added as out-of-shortlist words.

    If the state does not contain data and no vocabulary is given, constructs a
    vocabulary that contains all the training set words. In that case,
    --num-classes argument can be used to control the number of classes.

    :type args: argparse.Namespace
    :param args: a collection of command line arguments

    :type state: hdf5.File
    :param state: HDF5 file where the vocabulary should be saved

    :rtype: Vocabulary
    :returns: the created vocabulary
    """

    if state.keys():
        print("Reading vocabulary from existing network state.")
        sys.stdout.flush()
        result = Vocabulary.from_state(state)
        if not result.has_unigram_probs():
            # This is for backward compatibility. Remove at some point.
            print("Computing unigram word probabilities from training set.")
            sys.stdout.flush()
            word_counts = compute_word_counts(args.training_set)
            shortlist_words = list(result.id_to_word)
            shortlist_set = set(shortlist_words)
            oos_words = [x for x in word_counts.keys()
                         if x not in shortlist_set]
            result.id_to_word = numpy.asarray(shortlist_words + oos_words,
                                              dtype=object)
            result.word_to_id = {word: word_id
                                 for word_id, word in enumerate(result.id_to_word)}
            result.compute_probs(word_counts, update_class_probs=False)
            result.get_state(state)

    elif args.vocabulary is None:
        print("Constructing vocabulary from training set.")
        sys.stdout.flush()
        word_counts = compute_word_counts(args.training_set)
        result = Vocabulary.from_word_counts(word_counts, args.num_classes)
        result.get_state(state)

    else:
        print("Reading vocabulary from {}.".format(args.vocabulary))
        sys.stdout.flush()
        word_counts = compute_word_counts(args.training_set)
        oos_words = word_counts.keys()
        with open(args.vocabulary, 'rt', encoding='utf-8') as vocab_file:
            result = Vocabulary.from_file(vocab_file,
                                          args.vocabulary_format,
                                          oos_words=oos_words)

        if args.vocabulary_format == 'classes':
            print("Computing class membership probabilities and unigram "
                  "probabilities for out-of-shortlist words.")
            sys.stdout.flush()
            update_class_probs = True
        else:
            print("Computing unigram probabilities for out-of-shortlist words.")
            sys.stdout.flush()
            update_class_probs = False
        result.compute_probs(word_counts,
                             update_class_probs=update_class_probs)
        result.get_state(state)

    print("Number of words in vocabulary:", result.num_words())
    print("Number of words in shortlist:", result.num_shortlist_words())
    print("Number of word classes:", result.num_classes())
    return result

def train(args):
    """A function that performs the "theanolm train" command.

    :type args: argparse.Namespace
    :param args: a collection of command line arguments
    """

    numpy.random.seed(args.random_seed)

    log_file = args.log_file
    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        print("Invalid logging level requested:", args.log_level)
        sys.exit(1)
    log_format = '%(asctime)s %(funcName)s: %(message)s'
    if args.log_file == '-':
        logging.basicConfig(stream=sys.stdout, format=log_format, level=log_level)
    else:
        logging.basicConfig(filename=log_file, format=log_format, level=log_level)

    if args.debug:
        theano.config.compute_test_value = 'warn'
        print("Enabled computing test values for tensor variables.")
        print("Warning: GpuArray backend will fail random number generation!")
    else:
        theano.config.compute_test_value = 'off'
    theano.config.profile = args.profile
    theano.config.profile_memory = args.profile

    with h5py.File(args.model_path, 'a', driver='core') as state:
        vocabulary = _read_vocabulary(args, state)

        if args.num_noise_samples > vocabulary.num_classes():
            print("Number of noise samples ({}) is larger than the number of "
                  "classes. This doesn't make sense and would cause unigram "
                  "sampling to fail.".format(args.num_noise_samples))
            sys.exit(1)

        num_training_files = len(args.training_set)
        if len(args.weights) > num_training_files:
            print("You specified more weights than training files.")
            sys.exit(1)
        weights = numpy.ones(num_training_files).astype(theano.config.floatX)
        for index, weight in enumerate(args.weights):
            weights[index] = weight

        training_options = {
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

        optimization_options = {
            'method': args.optimization_method,
            'epsilon': args.numerical_stability_term,
            'gradient_decay_rate': args.gradient_decay_rate,
            'sqr_gradient_decay_rate': args.sqr_gradient_decay_rate,
            'learning_rate': args.learning_rate,
            'weights': weights,
            'momentum': args.momentum,
            'max_gradient_norm': args.gradient_normalization,
            'num_noise_samples': args.num_noise_samples,
            'noise_sharing': args.noise_sharing,
        }
        logging.debug("OPTIMIZATION OPTIONS")
        for option_name, option_value in optimization_options.items():
            if isinstance(option_value, list):
                value_str = ', '.join(str(x) for x in option_value)
                logging.debug("%s: [%s]", option_name, value_str)
            else:
                logging.debug("%s: %s", option_name, str(option_value))

        if len(args.sampling) > len(args.training_set):
            print("You specified more sampling coefficients than training "
                  "files.")
            sys.exit(1)

        print("Creating trainer.")
        sys.stdout.flush()
        trainer = Trainer(training_options, vocabulary, args.training_set,
                          args.sampling)
        trainer.set_logging(args.log_interval)

        print("Building neural network.")
        sys.stdout.flush()
        if args.architecture == 'lstm300' or args.architecture == 'lstm1500':
            architecture = Architecture.from_package(args.architecture)
        else:
            with open(args.architecture, 'rt', encoding='utf-8') as arch_file:
                architecture = Architecture.from_description(arch_file)

        network = Network(architecture, vocabulary, trainer.class_prior_probs,
                          default_device=args.default_device,
                          profile=args.profile)

        network.set_sampling(args.noise_distribution, args.noise_dampening,
                             args.noise_sharing)

        print("Building optimizer.")
        sys.stdout.flush()
        exclude_id = vocabulary.word_to_id['<unk>'] if args.exclude_unk \
                     else None
        epsilon = args.numerical_stability_term
        if args.cost == 'cross-entropy':
            cost_function = CrossEntropyCost(network, exclude_id, epsilon)
        elif args.cost == 'nce':
            cost_function = NCECost(network, exclude_id, epsilon)
        elif args.cost == 'blackout':
            cost_function = BlackoutCost(network, exclude_id, epsilon)
        else:
            print("Invalid cost function requested: `{}'".format(args.cost))
            sys.exit(1)
        try:
            optimizer = create_optimizer(optimization_options, network,
                                         cost_function, profile=args.profile)
        except theano.gradient.DisconnectedInputError as e:
            print("Cannot train the neural network because some of the "
                  "parameters are disconnected from the output. Make sure all "
                  "the layers are correctly connected in the network "
                  "architecture. The error message was: `{}'".format(e))

        if args.print_graph:
            print("Cost function computation graph:")
            theano.printing.debugprint(optimizer.gradient_update_function)

        trainer.initialize(network, state, optimizer)

        if args.validation_file is not None:
            print("Building text scorer for cross-validation.")
            sys.stdout.flush()
            scorer = TextScorer(network, use_shortlist=True,
                                exclude_unk=args.exclude_unk,
                                profile=args.profile)
            print("Validation text:", args.validation_file.name)
            validation_mmap = mmap.mmap(args.validation_file.fileno(),
                                        0,
                                        prot=mmap.PROT_READ)
            validation_iter = \
                LinearBatchIterator(validation_mmap,
                                    vocabulary,
                                    batch_size=args.batch_size,
                                    max_sequence_length=args.sequence_length,
                                    map_oos_to_unk=False)
            trainer.set_validation(validation_iter, scorer)
        else:
            print("Cross-validation will not be performed.")
            validation_iter = None

        print("Training neural network.")
        sys.stdout.flush()
        trainer.train()

        if 'layers' not in state.keys():
            print("The model has not been trained. No cross-validations were "
                  "performed or training did not improve the model.")
        elif validation_iter is not None:
            network.set_state(state)
            perplexity = scorer.compute_perplexity(validation_iter)
            print("Best validation set perplexity:", perplexity)
