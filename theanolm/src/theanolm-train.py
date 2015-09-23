#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy
import theano
import theanolm
import theanolm.trainers as trainers
from filetypes import TextFileType

def save_training_state(path):
	"""Saves current neural network and training state to disk.

	:type path: str
	:param path: filesystem path where to save the parameters to
	"""

	state = rnnlm.get_state()
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

def train(rnnlm, trainer, scorer, sentence_starts, validation_iter, args):
	best_params = None

	while trainer.epoch_number <= args.max_epochs:
		initial_cost = scorer.negative_log_probability(validation_iter)
		print("Validation set average sentence cost at the start of epoch %d: %f" % (
				trainer.epoch_number,
				initial_cost))
		
		print("Creating a random permutation of %d training sentences." % len(sentence_starts))
		training_order = numpy.random.permutation(sentence_starts)
		training_iter = theanolm.OrderedBatchIterator(
				args.training_file, 
				dictionary,
				training_order,
				batch_size=args.batch_size,
				max_sequence_length=args.sequence_length)
		
		while trainer.update_minibatch(training_iter, args.learning_rate):
			if (args.verbose_interval >= 1) and \
			   (trainer.total_updates % args.verbose_interval == 0):
				trainer.print_update_stats()
	
			if (args.validation_interval >= 1) and \
			   (trainer.total_updates % args.validation_interval == 0):
				validation_cost = scorer.negative_log_probability(validation_iter)
				if numpy.isnan(validation_cost):
					print("Stopping because an invalid floating point operation was performed while computing validation set cost. (Gradients exploded or vanished?)")
					return best_params
				if numpy.isinf(validation_cost):
					print("Stopping because validation set cost exploded to infinity.")
					return best_params
	
				trainer.append_validation_cost(validation_cost)
				validations_since_best = trainer.validations_since_min_cost()
				if validations_since_best == 0:
					best_params = rnnlm.get_state()
				elif (args.wait_improvement >= 0) and \
				     (validations_since_best > args.wait_improvement):
					if validations_cost >= initial_cost:
						args.learning_rate /= 2
					break
	
			if (args.save_interval >= 1) and \
			   (trainer.total_updates % args.save_interval == 0):
				# Save the best parameters and the current state.
				if not best_params is None:
					save_model(args.model_path, best_params)
				save_training_state(args.state_path)

	print("Stopping because %d epochs was reached." % args.max_epochs)
	validation_cost = scorer.negative_log_probability(validation_iter)
	trainer.append_validation_cost(validation_cost)
	if trainer.validations_since_min_cost() == 0:
		best_params = rnnlm.get_state()
	return best_params


parser = argparse.ArgumentParser()

argument_group = parser.add_argument_group("training")
argument_group.add_argument('model_path', metavar='MODEL', type=str,
		help='path where the best model state will be saved in numpy .npz format')
argument_group.add_argument('training_file', metavar='TRAINING-SET', type=TextFileType('r'),
		help='text or .gz file containing training data (one sentence per line)')
argument_group.add_argument('validation_file', metavar='VALIDATION-SET', type=TextFileType('r'),
		help='text or .gz file containing validation data (one sentence per line) for early stopping')
argument_group.add_argument('dictionary_file', metavar='DICTIONARY', type=TextFileType('r'),
		help='text or .gz file containing word list (one word per line) or word to class ID mappings (word and ID per line)')
argument_group.add_argument('--training-state', dest='state_path', metavar='FILE', type=str, default=None,
		help='the last training state will be read from and written to FILE in numpy .npz format (if not given, starts from scratch and only saves the best model)')
argument_group.add_argument('--sequence-length', metavar='N', type=int, default=100,
		help='ignore sentences longer than N words (default 100)')
argument_group.add_argument('--batch-size', metavar='N', type=int, default=16,
		help='each mini-batch will contain N sentences (default 16)')
argument_group.add_argument('--wait-improvement', metavar='N', type=int, default=10,
		help='wait N updates for validation set cost to decrease before stopping; if less than zero, stops only after maximum number of epochs is reached (default 10)')
argument_group.add_argument('--max-epochs', metavar='N', type=int, default=1000,
		help='perform at most N training epochs (default 1000)')
argument_group.add_argument('--optimization-method', metavar='NAME', type=str, default='adam',
		help='optimization method, one of "sgd", "nesterov", "adadelta", "rmsprop-sgd", "rmsprop-momentum", "adam" (default "adam")')
argument_group.add_argument('--learning-rate', metavar='ALPHA', type=float, default=0.001,
		help='initial learning rate (default 0.001)')
argument_group.add_argument('--momentum', metavar='BETA', type=float, default=0.9,
		help='momentum coefficient for momentum optimization methods (default 0.9)')
argument_group.add_argument('--validation-interval', metavar='N', type=int, default=1000,
		help='cross-validation for early stopping is performed after every Nth mini-batch update (default 1000)')
argument_group.add_argument('--save-interval', metavar='N', type=int, default=1000,
		help='save training state after every Nth mini-batch update; if less than one, save the model only after training (default 1000)')
argument_group.add_argument('--verbose-interval', metavar='N', type=int, default=100,
		help='print statistics of every Nth mini-batch update; quiet if less than one (default 100)')

argument_group = parser.add_argument_group("network structure")
argument_group.add_argument('--word-projection-dim', metavar='N', type=int, default=100,
		help='word projections will be N-dimensional (default 100)')
argument_group.add_argument('--hidden-layer-size', metavar='N', type=int, default=1000,
		help='hidden layer will contain N neurons (default 1000)')
argument_group.add_argument('--hidden-layer-type', metavar='NAME', type=str, default='lstm',
		help='hidden layer unit type, "lstm" or "gru" (default "lstm")')

argument_group = parser.add_argument_group("debugging")
argument_group.add_argument('--debug', metavar='\b', type=bool, default=False,
		help='enables debugging Theano errors')
argument_group.add_argument('--profile', metavar='\b', type=bool, default=False,
		help='enables profiling Theano functions')

args = parser.parse_args()

theano.config.compute_test_value = 'warn' if args.debug else 'off'

if (not args.state_path is None) and os.path.exists(args.state_path):
	print("Reading previous state from %s." % args.state_path)
	initial_state = numpy.load(args.state_path)
else:
	initial_state = None

print("Reading dictionary.")
dictionary = theanolm.Dictionary(args.dictionary_file)
print("Number of words in vocabulary:", dictionary.num_words())
print("Number of word classes:", dictionary.num_classes())

print("Finding sentence start positions in training data.")
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
rnnlm = theanolm.RNNLM(
		dictionary,
		args.word_projection_dim,
		args.hidden_layer_type,
		args.hidden_layer_size,
		args.profile)
if not initial_state is None:
	print("Restoring neural network to previous state.")
	rnnlm.set_state(initial_state)

print("Building neural network trainer.")
if args.optimization_method == 'sgd':
	trainer = trainers.SGDTrainer(rnnlm, args.profile)
elif args.optimization_method == 'nesterov':
	trainer = trainers.NesterovTrainer(rnnlm, args.momentum, args.profile)
elif args.optimization_method == 'adadelta':
	trainer = trainers.AdadeltaTrainer(rnnlm, args.profile)
elif args.optimization_method == 'rmsprop-sgd':
	trainer = trainers.RMSPropSGDTrainer(rnnlm, args.profile)
elif args.optimization_method == 'rmsprop-momentum':
	trainer = trainers.RMSPropMomentumTrainer(rnnlm, args.momentum, args.profile)
elif args.optimization_method == 'adam':
	trainer = trainers.AdamTrainer(rnnlm, args.profile)
else:
	print("Invalid optimization method requested:", args.optimization_method)
	exit(1)
if not initial_state is None:
	print("Restoring training to previous state.")
	trainer.set_state(initial_state)

print("Building text scorer.")
scorer = theanolm.TextScorer(rnnlm, args.profile)

print("Training neural network.")
best_params = train(rnnlm, trainer, scorer, sentence_starts, validation_iter, args)

print("Saving neural network and training state.")
save_training_state(args.state_path)
if best_params is None:
	print("Validation set cost did not decrease during training.")
else:
	save_model(args.model_path, best_params)
	rnnlm.set_state(best_params)
	validation_cost = scorer.negative_log_probability(validation_iter)
	print("Best validation set average sentence cost:", validation_cost)
