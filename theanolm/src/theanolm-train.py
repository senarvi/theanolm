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

def train(network, trainer, scorer, training_iter, validation_iter, args):
    network_state_min_cost = None
    trainer_state_min_cost = None

    while trainer.epoch_number <= args.max_epochs:
        initial_ppl = scorer.compute_perplexity(validation_iter)
        print("Validation set perplexity at the start of epoch {0}/{1}: {2}"
              "".format(trainer.epoch_number, args.max_epochs, initial_ppl))

        while trainer.update_minibatch(training_iter):
            if (args.verbose_interval >= 1) and \
               (trainer.total_updates % args.verbose_interval == 0):
                trainer.log_update()
                sys.stdout.flush()

            if (args.validation_interval >= 1) and \
               (trainer.total_updates % args.validation_interval == 0):
                validation_ppl = scorer.compute_perplexity(validation_iter)
                if numpy.isnan(validation_ppl):
                    print("Stopping because an invalid floating point "
                          "operation was performed while computing validation "
                          "set perplexity. (Gradients exploded?)")
                    return network_state_min_cost
                if numpy.isinf(validation_ppl):
                    print("Stopping because validation set perplexity exploded "
                          "to infinity.")
                    return network_state_min_cost

                trainer.append_validation_cost(validation_ppl)
                validations_since_best = trainer.validations_since_min_cost()
                if validations_since_best == 0:
                    # This is the minimum cost so far.
                    network_state_min_cost = network.get_state()
                    trainer_state_min_cost = trainer.get_state()
                elif (args.wait_improvement >= 0) and \
                     (validations_since_best > args.wait_improvement):
                    # Too many validations without improvement.
                    network.set_state(network_state_min_cost)  # XXX
                    trainer.set_state(trainer_state_min_cost)  # XXX
# XXX               if validation_ppl >= initial_ppl:
# XXX                   trainer.decrese_learning_rate(only_reset_cost_and_timestep)
# XXX               else:
# XXX                   trainer.decrease_learning_rate()
                    trainer.decrease_learning_rate()

            if (args.save_interval >= 1) and \
               (trainer.total_updates % args.save_interval == 0):
                # Save the best parameters and the current state.
                if not network_state_min_cost is None:
                    save_model(args.model_path, network_state_min_cost)
                save_training_state(args.state_path, network, trainer)

    print("Stopping because %d epochs was reached." % args.max_epochs)
    validation_ppl = scorer.compute_perplexity(validation_iter)
    trainer.append_validation_cost(validation_ppl)
    if trainer.validations_since_min_cost() == 0:
        network_state_min_cost = network.get_state()
    return network_state_min_cost


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

argument_group = parser.add_argument_group("training options")
argument_group.add_argument(
    '--sequence-length', metavar='N', type=int, default=100,
    help='ignore sentences longer than N words (default 100)')
argument_group.add_argument(
    '--batch-size', metavar='N', type=int, default=16,
    help='each mini-batch will contain N sentences (default 16)')
argument_group.add_argument(
    '--wait-improvement', metavar='N', type=int, default=10,
    help='wait N updates for validation set perplexity to decrease before stopping; '
         'if less than zero, stops only after maximum number of epochs is '
         'reached (default 10)')
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
    '--validation-interval', metavar='N', type=int, default=1000,
    help='cross-validation for early stopping is performed after every Nth '
         'mini-batch update (default 1000)')
argument_group.add_argument(
    '--save-interval', metavar='N', type=int, default=1000,
    help='save training state after every Nth mini-batch update; if less than '
         'one, save the model only after training (default 1000)')
argument_group.add_argument(
    '--verbose-interval', metavar='N', type=int, default=100,
    help='print statistics of every Nth mini-batch update; quiet if less than '
         'one (default 100)')

argument_group = parser.add_argument_group("debugging")
argument_group.add_argument(
    '--debug', action="store_true",
    help='enables debugging Theano errors')
argument_group.add_argument(
    '--profile', action="store_true",
    help='enables profiling Theano functions')

args = parser.parse_args()

logging.basicConfig(format="%(funcName)s: %(message)s")

if args.debug:
    logging.getLogger('root').setLevel(logging.DEBUG)
    theano.config.compute_test_value = 'warn'
else:
    logging.getLogger('root').setLevel(logging.INFO)
    theano.config.compute_test_value = 'off'

try:
    script_path = os.path.dirname(os.path.realpath(__file__))
    git_description = subprocess.check_output(['git', 'describe'], cwd=script_path)
    logging.info("Git repository description: %s", git_description.decode('utf-8'))
except CalledProcessError:
    logging.info("Git repository description is not available.")

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

training_iter = theanolm.ShufflingBatchIterator(
    args.training_file,
    dictionary,
    sentence_starts,
    batch_size=args.batch_size,
    max_sequence_length=args.sequence_length)

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
    args.skip_layer_size)
print(architecture)
network = theanolm.Network(dictionary, architecture, args.profile)
if not initial_state is None:
    print("Restoring neural network to previous state.")
    network.set_state(initial_state)

print("Building neural network trainer.")
sys.stdout.flush()
training_options = {
    'epsilon': args.numerical_stability_term,
    'gradient_decay_rate': args.gradient_decay_rate,
    'sqr_gradient_decay_rate': args.sqr_gradient_decay_rate,
    'learning_rate': args.learning_rate,
    'momentum': args.momentum}
if not args.gradient_normalization is None:
    training_options['max_gradient_norm'] = args.gradient_normalization
if args.optimization_method == 'sgd':
    trainer = trainers.SGDTrainer(network,
                                  training_options,
                                  args.profile)
elif args.optimization_method == 'nesterov':
    trainer = trainers.NesterovTrainer(network,
                                       training_options,
                                       args.profile)
elif args.optimization_method == 'adadelta':
    trainer = trainers.AdadeltaTrainer(network,
                                       training_options,
                                       args.profile)
elif args.optimization_method == 'rmsprop-sgd':
    trainer = trainers.RMSPropSGDTrainer(network,
                                         training_options,
                                         args.profile)
elif args.optimization_method == 'rmsprop-momentum':
    trainer = trainers.RMSPropMomentumTrainer(network,
                                              training_options,
                                              args.profile)
elif args.optimization_method == 'adam':
    trainer = trainers.AdamTrainer(network,
                                   training_options,
                                   args.profile)
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
best_state = train(network, trainer, scorer, training_iter, validation_iter, args)

print("Saving neural network and training state.")
sys.stdout.flush()
save_training_state(args.state_path, network, trainer)
if best_state is None:
    print("Validation set perplexity did not decrease during training.")
else:
    save_model(args.model_path, best_state)
    network.set_state(best_state)
    validation_ppl = scorer.compute_perplexity(validation_iter)
    print("Best validation set perplexity:", validation_ppl)
