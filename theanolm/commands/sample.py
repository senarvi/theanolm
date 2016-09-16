#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import subprocess
import numpy
import h5py
import theano
from theanolm import Vocabulary, Architecture, Network, TextSampler
from theanolm.filetypes import TextFileType

def add_arguments(parser):
    argument_group = parser.add_argument_group("files")
    argument_group.add_argument(
        'model_path', metavar='MODEL-FILE', type=str,
        help='the model file that will be used to generate text')
    argument_group.add_argument(
        '--output-file', metavar='FILE', type=TextFileType('w'), default='-',
        help='where to write the generated sentences (default stdout, will be '
             'compressed if the name ends in ".gz")')

    argument_group = parser.add_argument_group("sampling")
    argument_group.add_argument(
        '--num-sentences', metavar='N', type=int, default=10,
        help='generate N sentences')
    argument_group.add_argument(
        '--random-seed', metavar='N', type=int, default=None,
        help='seed to initialize the random state (default is to seed from a '
             'random source provided by the oprating system)')
    
    argument_group = parser.add_argument_group("debugging")
    argument_group.add_argument(
        '--debug', action="store_true",
        help='enables debugging Theano errors')

def sample(args):
    numpy.random.seed(args.random_seed)

    if args.debug:
        theano.config.compute_test_value = 'warn'
    else:
        theano.config.compute_test_value = 'off'

    with h5py.File(args.model_path, 'r') as state:
        print("Reading vocabulary from network state.")
        sys.stdout.flush()
        vocabulary = Vocabulary.from_state(state)
        print("Number of words in vocabulary:", vocabulary.num_words())
        print("Number of word classes:", vocabulary.num_classes())
        print("Building neural network.")
        sys.stdout.flush()
        architecture = Architecture.from_state(state)
        network = Network(vocabulary, architecture,
                          mode=Network.Mode.distribution)
        print("Restoring neural network state.")
        network.set_state(state)

    print("Building text sampler.")
    sys.stdout.flush()
    sampler = TextSampler(network)

    sequences = sampler.generate(30, args.num_sentences)
    for sequence in sequences:
        try:
            eos_pos = sequence.index('</s>')
            sequence = sequence[:eos_pos+1]
        except:
            pass
        args.output_file.write(' '.join(sequence) + '\n')
