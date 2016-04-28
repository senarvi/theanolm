#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import subprocess
import numpy
import h5py
import theano
from theanolm import Vocabulary, Architecture, Network, TextScorer
from theanolm import LinearBatchIterator
from theanolm.filetypes import TextFileType
from theanolm.iterators import utterance_from_line

def add_arguments(parser):
    argument_group = parser.add_argument_group("files")
    argument_group.add_argument(
        'model_path', metavar='MODEL-FILE', type=str,
        help='path where the best model state will be saved in numpy .npz '
             'format')
    argument_group.add_argument(
        'input_file', metavar='INPUT-FILE', type=TextFileType('r'),
        help='text or .gz file containing text to be scored (one sentence per '
             'line)')
    argument_group.add_argument(
        '--output-file', metavar='FILE', type=TextFileType('w'), default='-',
        help='where to write the statistics (default stdout)')
    
    argument_group = parser.add_argument_group("scoring")
    argument_group.add_argument(
        '--output', metavar='DETAIL', type=str, default='text',
        help='what to output, one of "perplexity", "utterance-scores", '
             '"word-scores" (default "perplexity")')
    argument_group.add_argument(
        '--log-base', metavar='B', type=int, default=None,
        help='convert output log probabilities to base B (default is the '
             'natural logarithm)')
    argument_group.add_argument(
        '--unk-penalty', metavar='LOGPROB', type=float, default=None,
        help="if LOGPROB is zero, do not include <unk> tokens in perplexity "
             "computation; otherwise use constant LOGPROB as <unk> token score "
             "(default is to use the network to predict <unk> probability)")

def score(args):
    with h5py.File(args.model_path, 'r') as state:
        print("Reading vocabulary from network state.")
        sys.stdout.flush()
        vocabulary = Vocabulary.from_state(state)
        print("Number of words in vocabulary:", vocabulary.num_words())
        print("Number of word classes:", vocabulary.num_classes())
        print("Building neural network.")
        sys.stdout.flush()
        architecture = Architecture.from_state(state)
        network = Network(vocabulary, architecture, batch_processing=True)
        print("Restoring neural network state.")
        sys.stdout.flush()
        network.set_state(state)

    print("Building text scorer.")
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
    scorer = TextScorer(network, ignore_unk, unk_penalty)

    print("Scoring text.")
    if args.output == 'perplexity':
        _score_text(args.input_file, vocabulary, scorer, args.output_file,
                    args.log_base, False)
    elif args.output == 'word-scores':
        _score_text(args.input_file, vocabulary, scorer, args.output_file,
                    args.log_base, True)
    elif args.output == 'utterance-scores':
        _score_utterances(args.input_file, vocabulary, scorer, args.output_file,
                          args.log_base)

def _score_text(input_file, vocabulary, scorer, output_file,
                log_base=None, word_level=False):
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

    :type word_level: bool
    :param word_level: if set to True, also writes word-level statistics
    """

    validation_iter = LinearBatchIterator(input_file, vocabulary)
    base_conversion = 1 if log_base is None else numpy.log(log_base)
    unk_id = vocabulary.word_to_id['<unk>']

    total_logprob = 0
    num_sentences = 0
    num_words = 0
    num_probs = 0
    for word_ids, _, mask in validation_iter:
        class_ids, membership_probs = vocabulary.get_class_memberships(word_ids)
        logprobs = scorer.score_batch(word_ids, class_ids, membership_probs,
                                      mask)
        for seq_index, seq_logprobs in enumerate(logprobs):
            seq_logprob = sum(seq_logprobs)
            num_probs += len(seq_logprobs)
            total_logprob += seq_logprob
            seq_word_ids = word_ids[:, seq_index]
            num_words += len(seq_word_ids)
            num_sentences += 1
            if not word_level:
                continue

            seq_logprobs = [x / base_conversion for x in seq_logprobs]
            seq_logprob /= base_conversion
            seq_class_names = vocabulary.word_ids_to_names(seq_word_ids)
            output_file.write("# Sentence {0}\n".format(num_sentences))

            # In case some word IDs are ignored, seq_word_ids may contain more
            # items than seq_logprobs.
            logprob_index = 0
            for word_index, word_id in enumerate(seq_word_ids[1:]):
                if word_index - 2 > 0:
                    history = seq_class_names[word_index:word_index - 3:-1]
                    history.append('...')
                else:
                    history = seq_class_names[word_index::-1]
                history = ', '.join(history)
                predicted = seq_class_names[word_index + 1]

                if scorer.ignore_unk and word_id == unk_id:
                    output_file.write("p({0} | {1}) is not predicted\n".format(
                        predicted, history))
                else:
                    logprob = seq_logprobs[logprob_index]
                    logprob_index += 1
                    output_file.write("log(p({0} | {1})) = {2}\n".format(
                        predicted, history, logprob))
            assert logprob_index == len(seq_logprobs)

            output_file.write("Sentence perplexity: {0}\n\n".format(
                numpy.exp(-seq_logprob / len(seq_logprobs))))

    output_file.write("Number of sentences: {0}\n".format(num_sentences))
    output_file.write("Number of words: {0}\n".format(num_words))
    output_file.write("Number of predicted probabilities: {0}\n".format(num_probs))
    if num_words > 0:
        cross_entropy = -total_logprob / num_probs
        perplexity = numpy.exp(cross_entropy)
        output_file.write("Cross entropy (base e): {0}\n".format(cross_entropy))
        if not log_base is None:
            cross_entropy /= base_conversion
            output_file.write("Cross entropy (base {1}): {0}\n".format(
                cross_entropy, log_base))
        output_file.write("Perplexity: {0}\n".format(perplexity))

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

    base_conversion = 1 if log_base is None else numpy.log(log_base)

    unk_id = vocabulary.word_to_id['<unk>']
    num_words = 0
    num_unks = 0
    for line_num, line in enumerate(input_file):
        words = utterance_from_line(line)
        if not words:
            continue

        word_ids = vocabulary.words_to_ids(words)
        num_words += word_ids.size
        num_unks += numpy.count_nonzero(word_ids == unk_id)
        class_ids = [vocabulary.word_id_to_class_id[word_id]
                     for word_id in word_ids]
        probs = [vocabulary.get_word_prob(word_id)
                 for word_id in word_ids]

        lm_score = scorer.score_sequence(word_ids, class_ids, probs)
        lm_score /= base_conversion
        output_file.write(str(lm_score) + '\n')

        if (line_num + 1) % 1000 == 0:
            print("{0} sentences scored.".format(line_num + 1))
        sys.stdout.flush()

    if num_words == 0:
        print("The input file contains no words.")
    else:
        print("{0} words processed, including start-of-sentence and "
              "end-of-sentence tags, and {1} ({2:.1f} %) out-of-vocabulary "
              "words".format(num_words, num_unks, num_unks / num_words))
