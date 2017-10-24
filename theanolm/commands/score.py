#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the "theanolm score" command.
"""

import sys
import logging

import numpy
import theano

from theanolm import Network
from theanolm.backend import TextFileType
from theanolm.parsing import ScoringBatchIterator
from theanolm.scoring import TextScorer

def add_arguments(parser):
    """Specifies the command line arguments supported by the "theanolm score"
    command.

    :type parser: argparse.ArgumentParser
    :param parser: a command line argument parser
    """

    argument_group = parser.add_argument_group("files")
    argument_group.add_argument(
        'model_path', metavar='MODEL-FILE', type=str,
        help='the model file that will be used to score text')
    argument_group.add_argument(
        'input_file', metavar='TEXT-FILE', type=TextFileType('r'),
        help='text file containing text to be scored (UTF-8, one sentence per '
             'line, assumed to be compressed if the name ends in ".gz")')
    argument_group.add_argument(
        '--output-file', metavar='FILE', type=TextFileType('w'), default='-',
        help='where to write the statistics (default stdout, will be '
             'compressed if the name ends in ".gz")')

    argument_group = parser.add_argument_group("scoring")
    argument_group.add_argument(
        '--output', metavar='DETAIL', type=str, default='perplexity',
        help='what to output, one of "perplexity", "utterance-scores", '
             '"word-scores" (default "perplexity")')
    argument_group.add_argument(
        '--log-base', metavar='B', type=int, default=None,
        help='convert output log probabilities to base B (default is the '
             'natural logarithm)')
    argument_group.add_argument(
        '--exclude-unk', action="store_true",
        help="exclude <unk> tokens from perplexity computation")
    argument_group.add_argument(
        '--subwords', metavar='MARKING', type=str, default=None,
        help='the subword vocabulary uses MARKING to indicate how words are '
             'formed from subwords; one of "word-boundary" (<w> token '
             'separates words), "prefix-affix" (subwords that can be '
             'concatenated are prefixed or affixed with +, e.g. "cat+ +s")')
    argument_group.add_argument(
        '--shortlist', action="store_true",
        help='distribute <unk> token probability among the out-of-shortlist '
             'words according to their unigram frequencies in the training '
             'data')

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
        help='use test values to get better error messages from Theano')
    argument_group.add_argument(
        '--profile', action="store_true",
        help='enable profiling Theano functions')

def score(args):
    """A function that performs the "theanolm score" command.

    :type args: argparse.Namespace
    :param args: a collection of command line arguments
    """

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
        print("Enabled computing test values for tensor variables.")
        print("Warning: GpuArray backend will fail random number generation!")
    else:
        theano.config.compute_test_value = 'off'
    theano.config.profile = args.profile
    theano.config.profile_memory = args.profile

    network = Network.from_file(args.model_path, exclude_unk=args.exclude_unk)

    print("Building text scorer.")
    sys.stdout.flush()
    scorer = TextScorer(network, args.shortlist, args.exclude_unk, args.profile)

    print("Scoring text.")
    if args.output == 'perplexity':
        _score_text(args.input_file, network.vocabulary, scorer,
                    args.output_file, args.log_base, args.subwords, False)
    elif args.output == 'word-scores':
        _score_text(args.input_file, network.vocabulary, scorer,
                    args.output_file, args.log_base, args.subwords, True)
    elif args.output == 'utterance-scores':
        _score_utterances(args.input_file, network.vocabulary, scorer,
                          args.output_file, args.log_base)
    else:
        print("Invalid output format requested:", args.output)
        sys.exit(1)

def _score_text(input_file, vocabulary, scorer, output_file,
                log_base=None, subword_marking=None, word_level=False):
    """Reads text from ``input_file``, computes perplexity using
    ``scorer``, and writes to ``output_file``.

    :type input_file: file object
    :param input_file: a file that contains the input sentences in SRILM n-best
                       format

    :type vocabulary: Vocabulary
    :param vocabulary: vocabulary that provides mapping between words and word
                       IDs

    :type scorer: TextScorer
    :param scorer: a text scorer for rescoring the input sentences

    :type output_file: file object
    :param output_file: a file where to write the output n-best list in SRILM
                        format

    :type log_base: int
    :param log_base: if set to other than None, convert log probabilities to
                     this base

    :type subword_marking: str
    :param subword_marking: if other than None, vocabulary is subwords;
        "word-boundary" indicates <w> token separates words, "prefix-affix"
        indicates subwords are prefixed/affixed with +

    :type word_level: bool
    :param word_level: if set to True, also writes word-level statistics
    """

    scoring_iter = \
        ScoringBatchIterator(input_file,
                             vocabulary,
                             batch_size=16,
                             max_sequence_length=None,
                             map_oos_to_unk=False)
    log_scale = 1.0 if log_base is None else numpy.log(log_base)

    total_logprob = 0.0
    num_sentences = 0
    num_tokens = 0
    num_words = 0
    num_probs = 0
    num_unks = 0
    num_zeroprobs = 0
    for word_ids, words, mask in scoring_iter:
        class_ids, membership_probs = vocabulary.get_class_memberships(word_ids)
        logprobs = scorer.score_batch(word_ids, class_ids, membership_probs,
                                      mask)
        for seq_index, seq_logprobs in enumerate(logprobs):
            seq_word_ids = word_ids[:, seq_index]
            seq_mask = mask[:, seq_index]
            seq_word_ids = seq_word_ids[seq_mask == 1]
            seq_words = words[seq_index]
            merged_words, merged_logprobs = _merge_subwords(seq_words,
                                                            seq_logprobs,
                                                            subword_marking)

            # total logprob of this sequence
            seq_logprob = sum(lp for lp in merged_logprobs
                              if (lp is not None) and (not numpy.isneginf(lp)))
            # total logprob of all sequences
            total_logprob += seq_logprob
            # number of tokens, which may be subwords, including <unk>'s
            num_tokens += len(seq_word_ids)
            # number of words, including <s>'s and <unk>'s
            num_words += len(merged_words)
            # number of word probabilities computed (may not include <unk>'s)
            num_seq_probs = sum((lp is not None) and (not numpy.isneginf(lp))
                                for lp in merged_logprobs)
            num_probs += num_seq_probs
            # number of unks and zeroprobs (just for reporting)
            num_unks += sum(lp is None for lp in merged_logprobs)
            num_zeroprobs += sum(numpy.isneginf(lp) for lp in merged_logprobs)
            # number of sequences
            num_sentences += 1

            if word_level:
                output_file.write("# Sentence {0}\n".format(num_sentences))
                _write_word_scores(vocabulary, merged_words, merged_logprobs,
                                   output_file, log_scale)
                output_file.write("Sentence perplexity: {0}\n\n".format(
                    numpy.exp(-seq_logprob / num_seq_probs)))

    output_file.write("Number of sentences: {0}\n".format(num_sentences))
    output_file.write("Number of words: {0}\n".format(num_words))
    output_file.write("Number of tokens: {0}\n".format(num_tokens))
    output_file.write("Number of predicted probabilities: {0}\n"
                      .format(num_probs))
    output_file.write("Number of excluded (OOV) words: {0}\n"
                      .format(num_unks))
    output_file.write("Number of zero probabilities: {0}\n"
                      .format(num_zeroprobs))
    if num_words > 0:
        cross_entropy = -total_logprob / num_probs
        perplexity = numpy.exp(cross_entropy)
        output_file.write("Cross entropy (base e): {0}\n".format(cross_entropy))
        if log_base is not None:
            cross_entropy /= log_scale
            output_file.write("Cross entropy (base {1}): {0}\n".format(
                cross_entropy, log_base))
        output_file.write("Perplexity: {0}\n".format(perplexity))

def _merge_subwords(subwords, subword_logprobs, marking):
    """Creates a word list from a subword list.

    :type subwords: list of strs
    :param subwords: list of vocabulary words, which may be subwords

    :type subword_logprobs: list of floats
    :param subword_logprobs: list of log probabilities for each word/subword
                             starting from the second one, containing ``None``
                             in place of any ignored <unk>'s

    :type marking: str
    :param marking: ``None`` for word vocabulary, otherwise the type of subword
                    marking used: "word-boundary" if a word boundary token (<w>)
                    is used and "prefix-affix" if subwords are prefixed/affixed
                    with + when they can be concatenated

    :rtype: tuple of two lists
    :returns: the first item is a list of words, where each word is either a
              string or a list of subword strings; the second item is a list of
              log probabilities for each word, starting from the second one,
              containing ``None`` in place of any ignored <unk>'s
    """

    if len(subword_logprobs) != len(subwords) - 1:
        raise ValueError("Number of logprobs should be exactly one less than "
                         "the number of words.")

    if marking is None:
        # Vocabulary is already words.
        return subwords, subword_logprobs

    words = [[subwords[0]]]
    logprobs = []
    current_word = []
    current_logprob = 0.0

    if marking == 'word-boundary':
        # Words are separated by <w>. Merge subwords and logprobs between <w>
        # tokens. If any part of a word is <unk>, the whole word is <unk>.
        for subword, logprob in zip(subwords[1:], subword_logprobs):
            if (current_logprob is None) or (logprob is None):
                current_logprob = None
            else:
                current_logprob += logprob
            if subword == '<w>':
                if current_word:
                    words.append(current_word)
                    logprobs.append(current_logprob)
                    current_word = []
                current_logprob = 0.0
            elif ('<unk>' in current_word) or (subword == '<unk>'):
                current_word = ['<unk>']
            else:
                current_word.append(subword)
    elif marking == 'prefix-affix':
        # Merge subword to the current word if the current word ends in + and
        # the subword starts with +.
        for subword, logprob in zip(subwords[1:], subword_logprobs):
            if current_word and current_word[-1].endswith('+') and \
               subword.startswith('+'):
                current_word.append(subword)
                if (current_logprob is None) or (logprob is None):
                    current_logprob = None
                else:
                    current_logprob += logprob
            else:
                if current_word:
                    words.append(current_word)
                    logprobs.append(current_logprob)
                current_word = [subword]
                current_logprob = logprob
    else:
        raise ValueError("Invalid subword marking type: " + marking)

    if current_word:
        words.append(current_word)
        logprobs.append(current_logprob)
    return words, logprobs

def _write_word_scores(vocabulary, words, logprobs, output_file, log_scale):
    """Writes word-level scores to an output file.

    :type vocabulary: Vocabulary
    :param vocabulary: vocabulary for printing information about which word are
                       predicted by the network

    :type words: list
    :param words: either word strings or one list for each word that contains
                  the subwords

    :type logprobs: list of floats
    :param logprobs: logprob of each word starting from the second word

    :type output_file: file object
    :param output_file: a file where to write the output

    :type log_scale: float
    :param log_scale: divide logprobs by this amount to convert to correct base
    """

    def word_info(vocabulary, word):
        """Returns information whether a word is in the vocabulary and in the
        shortlist.
        """
        if not word in vocabulary:
            return word + ':OOV'
        word_id = vocabulary.word_to_id[word]
        if vocabulary.in_shortlist(word_id):
            return word + ':shortlist'
        else:
            return word + ':OOS'

    if len(logprobs) != len(words) - 1:
        raise ValueError("Number of logprobs should be exactly one less than "
                         "the number of words.")

    logprobs = [None if x is None else x / log_scale for x in logprobs]
    for index, logprob in enumerate(logprobs):
        if index - 2 > 0:
            history_list = ['...']
            history_list.extend(words[index - 2:index + 1])
        else:
            history_list = words[:index + 1]
        history_list = [' '.join(word) if isinstance(word, list) else word
                        for word in history_list]
        history = ' '.join(history_list)

        predicted = words[index + 1]
        if isinstance(predicted, list):
            info = ' '.join(word_info(vocabulary, subword)
                            for subword in predicted)
            predicted = ' '.join(predicted)
        else:
            info = word_info(vocabulary, predicted)

        if logprob is None:
            output_file.write("p({0} | {1}) is not predicted  [{2}]\n".format(
                predicted, history, info))
        else:
            output_file.write("log(p({0} | {1})) = {2}  [{3}]\n".format(
                predicted, history, logprob, info))

def _score_utterances(input_file, vocabulary, scorer, output_file,
                      log_base=None):
    """Reads utterances from ``input_file``, computes LM scores using
    ``scorer``, and writes one score per line to ``output_file``.

    Start-of-sentence and end-of-sentece tags (``<s>`` and ``</s>``) will be
    inserted at the beginning and the end of each utterance, if they're missing.
    Empty lines will be ignored, instead of interpreting them as the empty
    sentence ``<s> </s>``.

    :type input_file: file object
    :param input_file: a file that contains the input sentences in SRILM n-best
                       format

    :type vocabulary: Vocabulary
    :param vocabulary: vocabulary that provides mapping between words and word
                       IDs

    :type scorer: TextScorer
    :param scorer: a text scorer for rescoring the input sentences

    :type output_file: file object
    :param output_file: a file where to write the output n-best list in SRILM
                        format

    :type log_base: int
    :param log_base: if set to other than None, convert log probabilities to
                     this base
    """

    log_scale = 1.0 if log_base is None else numpy.log(log_base)

    for line_num, line in enumerate(input_file):
        lm_score = scorer.score_line(line, vocabulary)
        if lm_score is None:
            continue
        lm_score /= log_scale
        output_file.write(str(lm_score) + '\n')
        if (line_num + 1) % 1000 == 0:
            print("{0} sentences scored.".format(line_num + 1))
        sys.stdout.flush()

    if scorer.num_words == 0:
        print("The input file contains no words.")
    else:
        print("{0} words processed, including start-of-sentence and "
              "end-of-sentence tags, and {1} ({2:.1f} %) out-of-vocabulary "
              "words".format(scorer.num_words,
                             scorer.num_unks,
                             scorer.num_unks / scorer.num_words))
