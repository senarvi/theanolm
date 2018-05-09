#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the "theanolm sample" command.
"""

import sys

import numpy
import h5py
import theano
import logging

from theanolm import Vocabulary, Architecture, Network, TextSampler
from theanolm.backend import TextFileType, get_default_device

def add_arguments(parser):
    """Specifies the command line arguments supported by the "theanolm sample"
    command.

    :type parser: argparse.ArgumentParser
    :param parser: a command line argument parser
    """

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
    argument_group.add_argument(
        '--sentence-length', metavar='N', type=int, default=30,
        help='generate sentences of N tokens')
    argument_group.add_argument(
        '--seed-sequence', metavar='SEQUENCE', type=str,
        help='Use SEQUENCE as seed; ie. first compute forward passes with the sequence, then generate')

    argument_group = parser.add_argument_group("configuration")
    argument_group.add_argument(
        '--default-device', metavar='DEVICE', type=str, default=None,
        help='when multiple GPUs are present, use DEVICE as default')

    argument_group = parser.add_argument_group("debugging")
    argument_group.add_argument(
        '--debug', action="store_true",
        help='enables debugging Theano errors')

def sample(args):
    """A function that performs the "theanolm sample" command.

    :type args: argparse.Namespace
    :param args: a collection of command line arguments
    """

    numpy.random.seed(args.random_seed)

    if args.debug:
        theano.config.compute_test_value = 'warn'
    else:
        theano.config.compute_test_value = 'off'

    with h5py.File(args.model_path, 'r') as state:
        logging.info("Reading vocabulary from network state.")
        vocabulary = Vocabulary.from_state(state)
        logging.info("Number of words in vocabulary: %d",
                     vocabulary.num_words())
        logging.info("Number of words in shortlist: %d",
                     vocabulary.num_shortlist_words())
        logging.info("Number of word classes: %d",
                     vocabulary.num_classes())
        logging.info("Building neural network.")
        architecture = Architecture.from_state(state)
        default_device = get_default_device(args.default_device)
        network = Network(architecture, vocabulary, mode=Network.Mode(minibatch=False),
                          default_device=default_device)
        logging.info("Restoring neural network state.")
        network.set_state(state)

    logging.info("Building text sampler.")
    sampler = TextSampler(network)

    sequences = sampler.generate(args.sentence_length, args.num_sentences, seed_sequence=args.seed_sequence)
    for sequence in sequences:
        try:
            eos_pos = sequence.index('</s>')
            sequence = sequence[:eos_pos+1]
        except ValueError:
            pass
        args.output_file.write(' '.join(sequence) + '\n')
