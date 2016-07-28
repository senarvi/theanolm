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
        help="language model probabilities given by the model read from "
             "MODEL-FILE will be weighted by LAMBDA, when interpolating with "
             "the language model probabilities in the lattice (default is 1.0, "
             "meaning that the LM probabilities in the lattice will be "
             "ignored)")
    argument_group.add_argument(
        '--lm-scale', metavar='LMSCALE', type=float, default=None,
        help="scale language model log probabilities by LMSCALE when computing "
             "the total probability of a path (default is to use the LM scale "
             "specified in the lattice file, or 1.0 if not specified)")
    argument_group.add_argument(
        '--wi-penalty', metavar='WIP', type=float, default=None,
        help="penalize word insertion by adding WIP to the total log "
             "probability as many times as there are words in the path "
             "(without scaling WIP by LMSCALE)")
    argument_group.add_argument(
        '--log-base', metavar='B', type=int, default=None,
        help="convert output log probabilities to base B and WIP from base B "
             "(default is natural logarithm; this does not affect reading "
             "lattices, since they specify their internal log base)")
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

    log_scale = 1.0 if log_base is None else numpy.log(args.log_base)

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
    wi_penalty = args.wi_penalty * log_scale
    decoder = LatticeDecoder(network,
                             nnlm_weight=args.nnlm_weight,
                             lm_scale=args.lm_scale,
                             wi_penalty=wi_penalty
                             ignore_unk=ignore_unk,
                             unk_penalty=unk_penalty)

    for lattice_file in args.lattices:
        print("Reading word lattice.")
        lattice = SLFLattice(lattice_file)
        print("Decoding word lattice.")
        tokens = decoder.decode(lattice)
        best_token = tokens[0]
        logprob = best_token.total_logprob / log_scale
        words = vocabulary.id_to_word[token.history]
        output_file.write("{} {}\n".format(logprob, ' '.join(words))
