#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy
import theano
import theanolm
from filetypes import TextFileType

parser = argparse.ArgumentParser()

argument_group = parser.add_argument_group("files")
argument_group.add_argument(
    'model_path', metavar='MODEL', type=str,
    help='path where the best model state will be saved in numpy .npz format')
argument_group.add_argument(
    'validation_file', metavar='TEXT', type=TextFileType('r'),
    help='text or .gz file containing text to be scored (one sentence per line)')
argument_group.add_argument(
    'dictionary_file', metavar='DICTIONARY', type=TextFileType('r'),
    help='text or .gz file containing word list (one word per line) or word to class ID mappings (word and ID per line)')
argument_group.add_argument(
    '--dictionary-format', metavar='NAME', type=str, default=None,
    help='dictionary format, one of "words" (one word per line, default), "classes" (word and class ID per line), "srilm-classes" (class name, membership probability, and word per line)')

args = parser.parse_args()

state = numpy.load(args.model_path)

print("Reading dictionary.")
sys.stdout.flush()
dictionary = theanolm.Dictionary(args.dictionary_file, args.dictionary_format)
print("Number of words in vocabulary:", dictionary.num_words())
print("Number of word classes:", dictionary.num_classes())

validation_iter = theanolm.BatchIterator(
    args.validation_file,
    dictionary,
    batch_size=1,
    max_sequence_length=None)

print("Building neural network.")
sys.stdout.flush()
rnnlm = theanolm.RNNLM(
    dictionary,
    state['rnnlm_word_projection_dim'],
    state['rnnlm_hidden_layer_type'],
    state['rnnlm_hidden_layer_size'])
print("Restoring neural network state.")
rnnlm.set_state(state)

print("Building text scorer.")
sys.stdout.flush()
scorer = theanolm.TextScorer(rnnlm)

print("Average sentence negative log probability:", scorer.negative_log_probability(validation_iter))
