#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import sys
import logging

import theano

from theanolm import Network
from theanolm.backend import TextFileType
from theanolm.backend import get_default_device, log_free_mem
from theanolm.scoring import LatticeDecoder, KaldiLattice

def add_arguments(parser):
    argument_group = parser.add_argument_group("files")
    argument_group.add_argument(
        'model_path', metavar='MODEL-FILE', type=str,
        help='the model file that will be used to decode the lattice')
    argument_group.add_argument(
        'lattices_in', metavar='LAT-FILE', type=TextFileType('r'),
        help='Kaldi lattice archive containing text CompactLattices'
    )
    argument_group.add_argument(
        'wordmap', metavar='FILE', type=TextFileType('r'),
        help='Kaldi word ID map (usually words.txt)'
    )
    argument_group.add_argument(
        'lattices_out', metavar='LAT-FILE', type=TextFileType('w'), default='-',
        help='Kaldi lattice archive containing text CompactLattices'
    )

    argument_group = parser.add_argument_group("decoding")
    argument_group.add_argument(
        '--output', metavar='FORMAT', type=str, default='ref',
        help='format of the output, one of "ref" (default, utterance ID '
             'followed by words), "trn" (words followed by utterance ID in '
             'parentheses), "full" (utterance ID, acoustic score, language '
             'score, and number of words, followed by words)')
    argument_group.add_argument(
        '--lm-scale', metavar='LMSCALE', type=float, default=15.0,
        help="scale language model log probabilities by LMSCALE when computing "
             "the total probability of a path (default is to use the LM scale "
             "specified in the lattice file, or 1.0 if not specified)")
    argument_group.add_argument(
        '--unk-penalty', metavar='LOGPROB', type=float, default=None,
        help="if LOGPROB is zero, do not include <unk> tokens in perplexity "
             "computation; otherwise use constant LOGPROB as <unk> token score "
             "(default is to use the network to predict <unk> probability)")

    argument_group = parser.add_argument_group("pruning")
    argument_group.add_argument(
        '--max-tokens-per-node', metavar='T', type=int, default=None,
        help="keep only at most T tokens at each node when decoding a lattice "
             "(default is no limit)")
    argument_group.add_argument(
        '--beam', metavar='B', type=float, default=None,
        help="prune tokens whose log probability is at least B smaller than "
             "the log probability of the best token at any given time (default "
             "is no beam pruning)")
    argument_group.add_argument(
        '--recombination-order', metavar='O', type=int, default=None,
        help="keep only the best token, when at least O previous words are "
             "identical (default is to recombine tokens only if the entire "
             "word history matches)")
    argument_group.add_argument(
        '--prune-relative', metavar='R', type=int, default=None,
        help="if set, tighten the beam and the max-tokens-per-node pruning "
             "linearly in the number of tokens in a node; those parameters "
             "will be divided by the number of tokens and multiplied by R")
    argument_group.add_argument(
        '--abs-min-max-tokens', metavar='T', type=float, default=30,
        help="if prune-extra-limit is used, do not tighten max-tokens-per-node "
             "further than this (default is 30)")
    argument_group.add_argument(
        '--abs-min-beam', metavar='B', type=float, default=150,
        help="if prune-extra-limit is used, do not tighten the beam further "
             "than this (default is 150)")

    argument_group = parser.add_argument_group("configuration")
    argument_group.add_argument(
        '--default-device', metavar='DEVICE', type=str, default=None,
        help='when multiple GPUs are present, use DEVICE as default')

    argument_group = parser.add_argument_group("logging and debugging")
    argument_group.add_argument(
        '--log-file', metavar='FILE', type=str, default='-',
        help='path where to write log file (default is standard output)')
    argument_group.add_argument(
        '--log-level', metavar='LEVEL', type=str, default='info',
        help='minimum level of events to log, one of "debug", "info", "warn" '
             '(default "info")')
    argument_group.add_argument(
        '--debug', action="store_true",
        help='enables debugging Theano errors')
    argument_group.add_argument(
        '--profile', action="store_true",
        help='enables profiling Theano functions')


def rescore(args):
    log_file = args.log_file
    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        print("Invalid logging level requested:", args.log_level, file=sys.stderr)
        sys.exit(1)
    log_format = '%(asctime)s %(funcName)s: %(message)s'
    if args.log_file == '-':
        logging.basicConfig(stream=sys.stdout, format=log_format, level=log_level)
    else:
        logging.basicConfig(filename=log_file, format=log_format, level=log_level)

    if args.debug:
        theano.config.compute_test_value = 'warn'
    else:
        theano.config.compute_test_value = 'off'
    theano.config.profile = args.profile
    theano.config.profile_memory = args.profile

    default_device = get_default_device(args.default_device)
    network = Network.from_file(args.model_path,
                                mode=Network.Mode(minibatch=False),
                                default_device=default_device)

    if args.unk_penalty is None:
        ignore_unk = False
        unk_penalty = None
    elif args.unk_penalty == 0:
        ignore_unk = True
        unk_penalty = None
    else:
        ignore_unk = False
        unk_penalty = args.unk_penalty
    decoding_options = {
        'nnlm_weight': 1.0,
        'lm_scale': args.lm_scale,
        'wi_penalty': None,
        'ignore_unk': ignore_unk,
        'unk_penalty': unk_penalty,
        'linear_interpolation': False,
        'max_tokens_per_node': args.max_tokens_per_node,
        'beam': args.beam,
        'recombination_order': args.recombination_order,
        'prune_relative': args.prune_relative,
        'abs_min_max_tokens': args.abs_min_max_tokens,
        'abs_min_beam': args.abs_min_beam
    }
    logging.debug("DECODING OPTIONS")
    for option_name, option_value in decoding_options.items():
        logging.debug("%s: %s", option_name, str(option_value))

    logging.info("Building word lattice decoder.")
    sys.stdout.flush()
    decoder = LatticeDecoder(network, decoding_options)

    word_to_id = {}
    for i, line in enumerate(args.wordmap):
        parts = line.split()
        if len(parts) != 2:
            print("Invalid word ID map file.", file=sys.stderr)
            sys.exit(1)
        word = parts[0]
        word_id = int(parts[1])
        word_to_id[word] = word_id

    id_to_word = [None] * len(word_to_id)
    for word, id in word_to_id.items():
        id_to_word[id] = word

    while True:
        utterance_id = args.lattices_in.readline()
        if not utterance_id:
            break  # end of file
        utterance_id = utterance_id.strip()
        if not utterance_id:
            continue  # empty line
        logging.info("Utterance `%s'", utterance_id)
        log_free_mem()

        lattice_lines = []
        while True:
            line = args.lattices_in.readline().strip()
            if not line:
                break  # empty line
            lattice_lines.append(line)

        lattice = KaldiLattice(lattice_lines, id_to_word)
        lattice.utterance_id = utterance_id
        final_tokens, recomb_tokens = decoder.decode(lattice)

        rescored_lattice = Lattice.from_decoder(lattice,
                                                final_tokens,
                                                recomb_tokens,
                                                network.vocabulary)
        rescored_lattice.write_kaldi(utterance_id,
                                     args.lattices_out,
                                     word_to_id)

        del rescored_lattice, final_tokens, recomb_tokens
        gc.collect()
