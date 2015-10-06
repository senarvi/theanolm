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
    'dictionary_file', metavar='DICTIONARY', type=TextFileType('r'),
    help='text or .gz file containing word list (one word per line) or word to class ID mappings (word and ID per line)')
argument_group.add_argument(
    '--dictionary-format', metavar='FORMAT', type=str, default='words',
    help='dictionary format, one of "words" (one word per line, default), "classes" (word and class ID per line), "srilm-classes" (class name, membership probability, and word per line)')
argument_group.add_argument(
    '--output-file', metavar='OUTPUT', type=TextFileType('w'), default='-',
    help='where to write the generated sentences (default stdout)')

argument_group = parser.add_argument_group("sampling")
argument_group.add_argument(
    '--num-sentences', metavar='N', type=int, default=10,
    help='generate N sentences')
argument_group.add_argument(
    '--random-seed', metavar='N', type=int, default=12345,
    help='seed to initialize the random state, between 1 and 2147462578 (default 12345)')

args = parser.parse_args()

state = numpy.load(args.model_path)

print("Reading dictionary.")
sys.stdout.flush()
dictionary = theanolm.Dictionary(args.dictionary_file, args.dictionary_format)
print("Number of words in vocabulary:", dictionary.num_words())
print("Number of word classes:", dictionary.num_classes())

print("Building neural network.")
sys.stdout.flush()
architecture = theanolm.Network.Architecture.from_state(state)
print(architecture)
network = theanolm.Network(dictionary, architecture)
print("Restoring neural network state.")
network.set_state(state)

print("Building text sampler.")
sys.stdout.flush()
sampler = theanolm.TextSampler(network, dictionary, args.random_seed)

for i in range(args.num_sentences):
    words = sampler.generate()
    args.output_file.write('{}: {}\n'.format(
        i, ' '.join(words)))
