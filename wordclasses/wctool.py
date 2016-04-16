#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import logging
from time import time
from theanolm.filetypes import TextFileType
from theanolm import Vocabulary
from wordclasses import TheanoBigramOptimizer, NumpyBigramOptimizer
from wordclasses import WordStatistics
from wordclasses.functions import is_scheduled

def save(optimizer, output_file, output_format):
    """Writes the current classes to a file.

    If the output file is seekable, first rewinds and truncates the file.

    :type optimizer: BigramOptimizer
    :param optimizer: save the current state of this optimizer

    :type output_file: file object
    :param output_file: a file or stream where to save the classes

    :type output_format: str
    :param output_format: either "classes" or "srilm-classes" - selects the
                          output file format
    """

    if output_file.seekable():
        output_file.seek(0)
        output_file.truncate()

    for word, class_id, prob in optimizer.words():
        if output_format == 'classes':
            output_file.write('{} {}\n'.format(word, class_id))
        elif output_format == 'srilm-classes':
            output_file.write('CLASS-{:05d} {} {}\n'.format(class_id, prob, word))

def main():
    parser = argparse.ArgumentParser(prog='wctool')

    argument_group = parser.add_argument_group("files")
    argument_group.add_argument(
        '--training-set', metavar='FILE', type=TextFileType('r'),
        nargs='+', required=True,
        help='text or .gz files containing training data (one sentence per '
             'line)')
    argument_group.add_argument(
        '--vocabulary', metavar='FILE', type=TextFileType('r'), default=None,
        help='text or .gz file containing a list of words to include in class '
             'forming, and possibly their initial classes')
    argument_group.add_argument(
        '--vocabulary-format', metavar='FORMAT', type=str, default='words',
        help='vocabulary format, one of "words" (one word per line, default), '
             '"classes" (word and class ID per line), "srilm-classes" (class '
             'name, membership probability, and word per line)')
    argument_group.add_argument(
        '--output-file', metavar='FILE', type=TextFileType('w'), default='-',
        help='where to write the word classes (default stdout)')
    argument_group.add_argument(
        '--output-format', metavar='FORMAT', type=str, default='srilm-classes',
        help='format of the output file, one of "classes" (word and class ID '
             'per line), "srilm-classes" (default; class name, membership '
             'probability, and word per line)')
    argument_group.add_argument(
        '--output-frequency', metavar='N', type=int, default='1',
        help='save classes N times per optimization iteration (default 1)')

    argument_group = parser.add_argument_group("optimization")
    argument_group.add_argument(
        '--num-classes', metavar='N', type=int, default=2000,
        help='number of classes to form, if vocabulary is not specified '
             '(default 2000)')
    argument_group.add_argument(
        '--method', metavar='NAME', type=str, default='bigram-theano',
        help='method for creating word classes, one of "bigram-theano", '
             '"bigram-numpy" (default "bigram-theano")')

    argument_group = parser.add_argument_group("logging and debugging")
    argument_group.add_argument(
        '--log-file', metavar='FILE', type=str, default='-',
        help='path where to write log file (default is standard output)')
    argument_group.add_argument(
        '--log-level', metavar='LEVEL', type=str, default='info',
        help='minimum level of events to log, one of "debug", "info", "warn" '
             '(default "info")')
    argument_group.add_argument(
        '--log-interval', metavar='N', type=int, default=1000,
        help='print statistics after every Nth word; quiet if less than one '
             '(default 1000)')

    args = parser.parse_args()

    log_file = args.log_file
    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError("Invalid logging level requested: " + args.log_level)
    log_format = '%(asctime)s %(funcName)s: %(message)s'
    if args.log_file == '-':
        logging.basicConfig(stream=sys.stdout, format=log_format, level=log_level)
    else:
        logging.basicConfig(filename=log_file, format=log_format, level=log_level)

    if args.vocabulary is None:
        vocabulary = Vocabulary.from_corpus(args.training_set,
                                            args.num_classes)
        for subset_file in args.training_set:
            subset_file.seek(0)
    else:
        vocabulary = Vocabulary.from_file(args.vocabulary,
                                          args.vocabulary_format)

    print("Number of words in vocabulary:", vocabulary.num_words())
    print("Number of word classes:", vocabulary.num_classes())
    print("Number of normal word classes:", vocabulary.num_normal_classes)

    logging.info("Reading word unigram and bigram statistics.")
    statistics = WordStatistics(args.training_set, vocabulary)

    if args.method == 'bigram-theano':
        optimizer = TheanoBigramOptimizer(statistics, vocabulary)
    elif args.method == 'bigram-numpy':
        optimizer = NumpyBigramOptimizer(statistics, vocabulary)
    else:
        raise ValueError("Invalid method requested: " + args.method)

    iteration = 1
    while True:
        logging.info("Starting iteration %d.", iteration)
        num_words = 0
        num_moves = 0
        for word in vocabulary.words():
            start_time = time()
            num_words += 1
            if optimizer.move_to_best_class(word):
                num_moves += 1
            duration = time() - start_time
            if (args.log_interval >= 1) and \
               (num_words % args.log_interval == 0):
                logging.info("[%d] (%.1f %%) of iteration %d -- moves = %d, cost = %.2f, duration = %.1f ms",
                     num_words,
                     num_words / vocabulary.num_words() * 100,
                     iteration,
                     num_moves,
                     optimizer.log_likelihood(),
                     duration * 100)
            if is_scheduled(num_words,
                            args.output_frequency,
                            vocabulary.num_words()):
                save(optimizer, args.output_file, args.output_format)

        if num_moves == 0:
            break
        iteration += 1

    logging.info("Optimization finished.")
    save(optimizer, args.output_file, args.output_format)
