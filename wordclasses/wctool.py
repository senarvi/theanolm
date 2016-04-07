#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from theanolm.filetypes import TextFileType
from theanolm import Vocabulary
from wordclasses import TheanoBigramOptimizer, NumpyBigramOptimizer, WordStatistics

def main():
    parser = argparse.ArgumentParser(prog='wctool')
    parser.add_argument(
        '--training-set', metavar='TRAINING-SET', type=TextFileType('r'),
        nargs='+',
        help='text or .gz files containing training data (one sentence per '
             'line)')
    parser.add_argument(
        '--num-classes', metavar='N', type=int, default=2000,
        help='number of classes to form, if vocabulary is not specified '
             '(default 2000)')
    parser.add_argument(
        '--vocabulary', metavar='VOCAB', type=TextFileType('r'), default=None,
        help='text or .gz file containing a list of words to include in class '
             'forming, and possibly their initial classes')
    parser.add_argument(
        '--vocabulary-format', metavar='FORMAT', type=str, default='words',
        help='vocabulary format, one of "words" (one word per line, default), '
             '"classes" (word and class ID per line), "srilm-classes" (class '
             'name, membership probability, and word per line)')
    parser.add_argument(
        '--method', metavar='NAME', type=str, default='bigram-theano',
        help='method for creating word classes, one of "bigram-theano", '
             '"bigram-numpy" (default "bigram-theano")')
    parser.add_argument(
        '--output-format', metavar='FORMAT', type=str, default='srilm-classes',
        help='format of the output file, one of "classes" (word and class ID '
             'per line), "srilm-classes" (default; class name, membership '
             'probability, and word per line)')
    parser.add_argument(
        '--output-file', metavar='OUTPUT', type=TextFileType('w'), default='-',
        help='where to write the word classes (default stdout)')

    args = parser.parse_args()

    if args.vocabulary is None:
        vocabulary = Vocabulary.from_corpus(args.training_set,
                                            args.num_classes)
        for subset_file in args.training_set:
            subset_file.seek(0)
    else:
        vocabulary = Vocabulary.from_file(args.vocabulary,
                                          args.vocabulary_format)
    statistics = WordStatistics(args.training_set, vocabulary)

    if args.method == 'bigram-theano':
        optimizer = TheanoBigramOptimizer(statistics, vocabulary)
    elif args.method == 'bigram-numpy':
        optimizer = NumpyBigramOptimizer(statistics, vocabulary)
    else:
        raise ValueError("Invalid method requested: " + args.method)

    iteration = 1
    while True:
        print("Starting iteration {}.".format(iteration))
        num_words = 0
        num_moves = 0
        for word in vocabulary.words():
            num_words += 1
            if optimizer.move_to_best_class(word):
                num_moves += 1
                print("iteration {}, {} words, {} moves, log likelihood {}".format(
                    iteration, num_words, num_moves, optimizer.log_likelihood()))
        if num_moves == 0:
            break
        iteration += 1

    print("Optimization finished.")

    for word, class_id, prob in optimizer.words():
        if args.output_format == 'classes':
            args.output_file.write('{} {}\n'.format(word, class_id))
        elif args.output_format == 'srilm-classes':
            args.output_file.write('CLASS-{:05d} {} {}\n'.format(class_id, prob, word))
