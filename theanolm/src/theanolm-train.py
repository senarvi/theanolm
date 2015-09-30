#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import numpy
import theano
import theanolm
import theanolm.trainers as trainers
from filetypes import TextFileType

def save_training_state(path, network, trainer):
    """Saves current neural network and training state to disk.

    :type path: str
    :param path: filesystem path where to save the parameters to

    :type network: Network
    """

    state = network.get_state()
    state.update(trainer.get_state())
    numpy.savez(path, **state)
    print("Saved %d parameters to %s." % (len(state), path))

def save_model(path, state):
    """Saves the given model parameters to disk.

    :type path: str
    :param path: filesystem path where to save the parameters to

    :type state: dict
    :param state: dictionary of parameters to save
    """

    numpy.savez(path, **state)
    print("Saved %d parameters to %s." % (len(state), path))

def train(network, trainer, scorer, sentence_starts, validation_iter, args):
    best_params = None

    while trainer.epoch_number <= args.max_epochs:
        initial_cost = -scorer.score_text(validation_iter)
        print("Validation set average sentence cost at the start of epoch %d/%d: %f" % (
            trainer.epoch_number,
            args.max_epochs,
            initial_cost))

        print("Creating a random permutation of %d training sentences." % len(sentence_starts))
        sys.stdout.flush()
        numpy.random.shuffle(sentence_starts)
        training_iter = theanolm.OrderedBatchIterator(
            args.training_file,
            dictionary,
            sentence_starts,
            batch_size=args.batch_size,
            max_sequence_length=args.sequence_length)

        while trainer.update_minibatch(training_iter, args.learning_rate):
            if (args.verbose_interval >= 1) and \
               (trainer.total_updates % args.verbose_interval == 0):
                trainer.print_update_stats()
                sys.stdout.flush()

            if (args.validation_interval >= 1) and \
               (trainer.total_updates % args.validation_interval == 0):
                validation_cost = -scorer.score_text(validation_iter)
                if numpy.isnan(validation_cost):
                    print("Stopping because an invalid floating point operation was performed while computing validation set cost. (Gradients exploded or vanished?)")
                    return best_params
                if numpy.isinf(validation_cost):
                    print("Stopping because validation set cost exploded to infinity.")
                    return best_params

                trainer.append_validation_cost(validation_cost)
                trainer.print_cost_history()
                sys.stdout.flush()
                validations_since_best = trainer.validations_since_min_cost()
                if validations_since_best == 0:
                    best_params = network.get_state()
                elif (args.wait_improvement >= 0) and \
                     (validations_since_best > args.wait_improvement):
#                   if validation_cost >= initial_cost:
                    args.learning_rate /= 2
                    network.set_state(best_params)
                    trainer.next_epoch()
                    break

            if (args.save_interval >= 1) and \
               (trainer.total_updates % args.save_interval == 0):
                # Save the best parameters and the current state.
                if not best_params is None:
                    save_model(args.model_path, best_params)
                save_training_state(args.state_path, network, trainer)

    print("Stopping because %d epochs was reached." % args.max_epochs)
    validation_cost = -scorer.score_text(validation_iter)
    trainer.append_validation_cost(validation_cost)
    if trainer.validations_since_min_cost() == 0:
        best_params = network.get_state()
    return best_params


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
    help='text or .gz file containing validation data (one sentence per line) for early stopping')
argument_group.add_argument(
    'dictionary_file', metavar='DICTIONARY', type=TextFileType('r'),
    help='text or .gz file containing word list (one word per line) or word to class ID mappings (word and ID per line)')
argument_group.add_argument(
    '--training-state', dest='state_path', metavar='FILE', type=str, default=None,
    help='the last training state will be read from and written to FILE in numpy .npz format (if not given, starts from scratch and only saves the best model)')
argument_group.add_argument(
    '--dictionary-format', metavar='FORMAT', type=str, default='words',
    help='dictionary format, one of "words" (one word per line, default), "classes" (word and class ID per line), "srilm-classes" (class name, membership probability, and word per line)')

argument_group = parser.add_argument_group("network structure")
argument_group.add_argument(
    '--word-projection-dim', metavar='N', type=int, default=100,
    help='word projections will be N-dimensional (default 100)')
argument_group.add_argument(
    '--hidden-layer-size', metavar='N', type=int, default=1000,
    help='hidden layer will contain N neurons (default 1000)')
argument_group.add_argument(
    '--hidden-layer-type', metavar='TYPE', type=str, default='lstm',
    help='hidden layer unit type, "lstm" or "gru" (default "lstm")')
argument_group.add_argument(
    '--include-skip-layer', action="store_true",
    help='include connections that skip the hidden layer')

argument_group = parser.add_argument_group("training options")
argument_group.add_argument(
    '--sequence-length', metavar='N', type=int, default=100,
    help='ignore sentences longer than N words (default 100)')
argument_group.add_argument(
    '--batch-size', metavar='N', type=int, default=16,
    help='each mini-batch will contain N sentences (default 16)')
argument_group.add_argument(
    '--wait-improvement', metavar='N', type=int, default=10,
    help='wait N updates for validation set cost to decrease before stopping; if less than zero, stops only after maximum number of epochs is reached (default 10)')
argument_group.add_argument(
    '--max-epochs', metavar='N', type=int, default=1000,
    help='perform at most N training epochs (default 1000)')
argument_group.add_argument(
    '--optimization-method', metavar='METHOD', type=str, default='adam',
    help='optimization method, one of "sgd", "nesterov", "adadelta", "rmsprop-sgd", "rmsprop-momentum", "adam" (default "adam")')
argument_group.add_argument(
    '--learning-rate', metavar='ALPHA', type=float, default=0.001,
    help='initial learning rate (default 0.001)')
argument_group.add_argument(
    '--momentum', metavar='BETA', type=float, default=0.9,
    help='momentum coefficient for momentum optimization methods (default 0.9)')
argument_group.add_argument(
    '--validation-interval', metavar='N', type=int, default=1000,
    help='cross-validation for early stopping is performed after every Nth mini-batch update (default 1000)')
argument_group.add_argument(
    '--save-interval', metavar='N', type=int, default=1000,
    help='save training state after every Nth mini-batch update; if less than one, save the model only after training (default 1000)')
argument_group.add_argument(
    '--verbose-interval', metavar='N', type=int, default=100,
    help='print statistics of every Nth mini-batch update; quiet if less than one (default 100)')

argument_group = parser.add_argument_group("debugging")
argument_group.add_argument(
    '--debug', action="store_true",
    help='enables debugging Theano errors')
argument_group.add_argument(
    '--profile', action="store_true",
    help='enables profiling Theano functions')

args = parser.parse_args()

theano.config.compute_test_value = 'warn' if args.debug else 'off'

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

print("Finding sentence start positions in training data.")
sys.stdout.flush()
sentence_starts = [0]
# Can't use readline() here, otherwise TextIOWrapper disables tell().
ch = args.training_file.read(1)
while ch != '':
    if ch == '\n':
        pos = args.training_file.tell()
        ch = args.training_file.read(1)
        if ch != '':
            sentence_starts.append(pos)
    else:
        ch = args.training_file.read(1)

validation_iter = theanolm.BatchIterator(
    args.validation_file,
    dictionary,
    batch_size=args.batch_size,
    max_sequence_length=args.sequence_length)

print("Building neural network.")
sys.stdout.flush()
architecture = theanolm.Network.Architecture(
    args.word_projection_dim,
    args.hidden_layer_type,
    args.hidden_layer_size,
    args.include_skip_layer)
print(architecture)
network = theanolm.Network(dictionary, architecture, args.profile)
if not initial_state is None:
    print("Restoring neural network to previous state.")
    network.set_state(initial_state)

print("Building neural network trainer.")
sys.stdout.flush()
if args.optimization_method == 'sgd':
    trainer = trainers.SGDTrainer(network, args.profile)
elif args.optimization_method == 'nesterov':
    trainer = trainers.NesterovTrainer(network, args.momentum, args.profile)
elif args.optimization_method == 'adadelta':
    trainer = trainers.AdadeltaTrainer(network, args.profile)
elif args.optimization_method == 'rmsprop-sgd':
    trainer = trainers.RMSPropSGDTrainer(network, args.profile)
elif args.optimization_method == 'rmsprop-momentum':
    trainer = trainers.RMSPropMomentumTrainer(network, args.momentum, args.profile)
elif args.optimization_method == 'adam':
    trainer = trainers.AdamTrainer(network, args.profile)
else:
    print("Invalid optimization method requested:", args.optimization_method)
    exit(1)
if not initial_state is None:
    print("Restoring training to previous state.")
    trainer.set_state(initial_state)

print("Building text scorer.")
sys.stdout.flush()
scorer = theanolm.TextScorer(network, args.profile)

print("Training neural network.")
sys.stdout.flush()
best_params = train(network, trainer, scorer, sentence_starts, validation_iter, args)

print("Saving neural network and training state.")
sys.stdout.flush()
save_training_state(args.state_path, network, trainer)
if best_params is None:
    print("Validation set cost did not decrease during training.")
else:
    save_model(args.model_path, best_params)
    network.set_state(best_params)
    validation_logprob = scorer.score_text(validation_iter)
    print("Best validation set average sentence logprob:", validation_logprob)
