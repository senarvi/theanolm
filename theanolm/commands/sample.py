#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import subprocess
import numpy
import h5py
import theano
import theanolm
from theanolm.filetypes import TextFileType

def add_arguments(parser):
    argument_group = parser.add_argument_group("files")
    argument_group.add_argument(
        'model_path', metavar='MODEL', type=str,
        help='path where the best model state will be saved in numpy .npz '
             'format')
    argument_group.add_argument(
        'dictionary_file', metavar='DICTIONARY', type=TextFileType('r'),
        help='text or .gz file containing word list or class definitions')
    argument_group.add_argument(
        '--dictionary-format', metavar='FORMAT', type=str, default='words',
        help='dictionary format, one of "words" (one word per line, default), '
             '"classes" (word and class ID per line), "srilm-classes" (class '
             'name, membership probability, and word per line)')
    argument_group.add_argument(
        '--output-file', metavar='OUTPUT', type=TextFileType('w'), default='-',
        help='where to write the generated sentences (default stdout)')
    
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

    try:
        script_path = os.path.dirname(os.path.realpath(__file__))
        git_description = subprocess.check_output(['git', 'describe'], cwd=script_path)
        print("TheanoLM", git_description.decode('utf-8'))
    except subprocess.CalledProcessError:
        pass

    print("Reading dictionary.")
    sys.stdout.flush()
    dictionary = theanolm.Dictionary(args.dictionary_file, args.dictionary_format)
    print("Number of words in vocabulary:", dictionary.num_words())
    print("Number of word classes:", dictionary.num_classes())

    print("Building neural network.")
    sys.stdout.flush()
    with h5py.File(args.model_path, 'r') as state:
        architecture = theanolm.Network.Architecture.from_state(state)
        network = theanolm.Network(dictionary, architecture, batch_processing=False)
        print("Restoring neural network state.")
        network.set_state(state)

    print("Building text sampler.")
    sys.stdout.flush()
    sampler = theanolm.TextSampler(network, dictionary)

    for i in range(args.num_sentences):
        words = sampler.generate()
        args.output_file.write('{}: {}\n'.format(
            i, ' '.join(words)))
