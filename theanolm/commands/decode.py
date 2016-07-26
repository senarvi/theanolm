#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import subprocess
import numpy
import h5py
import theano
from theanolm import Vocabulary, Architecture, Network
from theanolm.scoring import LatticeDecoder
from theanolm.filetypes import TextFileType
from theanolm.iterators import utterance_from_line

def add_arguments(parser):
    argument_group = parser.add_argument_group("files")
    argument_group.add_argument(
        'model_path', metavar='MODEL-FILE', type=str,
        help='the model file that will be used to decode the lattice')
    argument_group.add_argument(
        '--lattices', metavar='FILE', type=TextFileType('r'), nargs='+',
        required=True,
        help='word lattices to be decoded (SLF, assumed to be compressed if '
             'the name ends in ".gz")')
    argument_group.add_argument(
        '--output-file', metavar='FILE', type=TextFileType('w'), default='-',
        help='where to write the best paths through the lattices (default '
             'stdout, will be compressed if the name ends in ".gz")')

    argument_group = parser.add_argument_group("decoding")
    argument_group.add_argument(
        '--nnlm-weight', metavar='LAMBDA', type=float, default=1.0,
        help='language model probabilities given by the model read from '
             'MODEL-FILE will be weighted by LAMBDA, when interpolating with '
             'the language model probabilities in the lattice (default is 1.0, '
             'meaning that the LM probabilities in the lattice will be '
             'ignored)')
    argument_group.add_argument(
        '--log-base', metavar='B', type=int, default=None,
        help='convert output log probabilities to base B (default is the '
             'natural logarithm)')
    argument_group.add_argument(
        '--unk-penalty', metavar='LOGPROB', type=float, default=None,
        help="if LOGPROB is zero, do not include <unk> tokens in perplexity "
             "computation; otherwise use constant LOGPROB as <unk> token score "
             "(default is to use the network to predict <unk> probability)")

def decode(args):
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
                          mode=Network.Mode.target_words)
        print("Restoring neural network state.")
        sys.stdout.flush()
        network.set_state(state)

    print("Building word lattice decoder.")
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
    decoder = LatticeDecoder(network, args.nnlm_weight, ignore_unk, unk_penalty)

    base_conversion = 1 if log_base is None else numpy.log(log_base)
    unk_id = vocabulary.word_to_id['<unk>']

    for lattice_file in args.lattices:
        print("Reading word lattice.")
        lattice = SLFLattice()
        lattice.read(lattice_file)
        print("Decoding word lattice.")
        decoder.decode(lattice)
        class_ids, membership_probs = vocabulary.get_class_memberships(word_ids)
        seq_logprobs = [x / base_conversion for x in seq_logprobs]
        seq_class_names = vocabulary.word_ids_to_names(seq_word_ids)
        # seq_logprob is in natural base.
        output_file.write("Sentence perplexity: {0}\n\n".format(
            numpy.exp(-seq_logprob / len(seq_logprobs))))
        if not log_base is None:
            cross_entropy /= base_conversion
            output_file.write("Cross entropy (base {1}): {0}\n".format(
                cross_entropy, log_base))
        output_file.write("Perplexity: {0}\n".format(perplexity))
