#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import logging
import theano
from theanolm import Network
from theanolm.scoring import LatticeDecoder, SLFLattice, KaldiLattice
from theanolm.filetypes import TextFileType


def add_arguments(parser):
    argument_group = parser.add_argument_group("files")
    argument_group.add_argument(
        'model_path', metavar='MODEL-FILE', type=str,
        help='the model file that will be used to decode the lattice')
    argument_group.add_argument(
        'lattices_in', metavar='LAT-FILE', type=TextFileType('r'),
        help='kaldi lattice archive containing text CompactLattices'
    )
    argument_group.add_argument(
        'wordmap', metavar='FILE', type=TextFileType('r'),
        help='kaldi words.txt'
    )
    argument_group.add_argument(
        'lattices_out', metavar='LAT-FILE', type=TextFileType('w'), default='-',
        help='kaldi lattice archive containing text CompactLattices'
    )

    argument_group = parser.add_argument_group("decoding")
    argument_group.add_argument(
        '--output', metavar='FORMAT', type=str, default='ref',
        help='format of the output, one of "ref" (default, utterance ID '
             'followed by words), "trn" (words followed by utterance ID in '
             'parentheses), "full" (utterance ID, acoustic score, language '
             'score, and number of words, followed by words)')
    argument_group.add_argument(
        '--n-best', metavar='N', type=int, default=1,
        help='print N best paths of each lattice (default 1)')
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
        print("Invalid logging level requested:", args.log_level)
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

    network = Network.from_file(args.model_path,
                                mode=Network.Mode(minibatch=False))

    log_scale = 1.0

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
        'recombination_order': args.recombination_order
    }
    logging.debug("DECODING OPTIONS")
    for option_name, option_value in decoding_options.items():
        logging.debug("%s: %s", option_name, str(option_value))

    print("Building word lattice decoder.")
    sys.stdout.flush()
    decoder = LatticeDecoder(network, decoding_options)

    word_to_id = {}

    for i, line in enumerate(args.wordmap):
        parts = line.split()
        assert len(parts) == 2
        word = parts[0]
        word_id = int(parts[1])
        word_to_id[word] = word_id
        assert i == word_id

    id_to_word = [None] * len(word_to_id)
    for word, id in word_to_id.items():
        id_to_word[id] = word

    while True:
        key = args.lattices_in.readline().strip()
        if len(key) == 0:
            break
        logging.info("Lat key: {}".format(key))
        lat_lines = []
        while True:
            line = args.lattices_in.readline().strip()
            if len(line) == 0:
                break
            lat_lines.append(line)

        lattice = KaldiLattice(lat_lines, id_to_word)
        lattice.utterance_id = key
        decoder.decode(lattice)


        decoder.write_kaldi(key, word_to_id, args.lattices_out)



def format_token(token, utterance_id, vocabulary, log_scale, output_format):
    """Formats an output line from a token and an utterance ID.

    Reads word IDs from the history list of ``token`` and converts them to words
    using ``vocabulary``. The history may contain also OOV words as text, so any
    ``str`` will be printed literally.

    :type token: Token
    :param token: a token whose history will be formatted

    :type utterance_id: str
    :param utterance_id: utterance ID for full output

    :type vocabulary: Vocabulary
    :param vocabulary: mapping from word IDs to words

    :type log_scale: float
    :param log_scale: divide log probabilities by this number to convert the log
                      base

    :type output_format: str
    :param output_format: which format to write, one of "ref" (utterance ID,
        words), "trn" (words, utterance ID in parentheses), "full" (utterance
        ID, acoustic and LM scores, number of words, words)

    :rtype: str
    :returns: the formatted output line
    """

    words = token.history_words(vocabulary)
    if output_format == 'ref':
        return "{} {}".format(utterance_id, ' '.join(words))
    elif output_format == 'trn':
        return "{} ({})".format(' '.join(words), utterance_id)
    elif output_format == 'full':
        return "{} {} {} {} {} {} {}".format(
            utterance_id,
            token.ac_logprob / log_scale,
            token.lm_logprob / log_scale,
            token.graph_logprob / log_scale,
            token.total_logprob / log_scale,
            len(words),
            ' '.join(words))
    else:
        print("Invalid output format requested:", args.output)
        sys.exit(1)
