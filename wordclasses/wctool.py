#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from theanolm.filetypes import TextFileType
from wordclasses import TheanoBigramOptimizer, NumpyBigramOptimizer

def main():
    parser = argparse.ArgumentParser(prog='wctool')
    parser.add_argument(
        'training_set', metavar='TRAINING-SET', type=TextFileType('r'),
        help='text or .gz files containing training data (one sentence per '
             'line)')
    parser.add_argument(
        '--num-classes', metavar='N', type=int, default=2000,
        help='number of classes to form (default 2000)')
    parser.add_argument(
        '--vocabulary', metavar='VOCABULARY', type=TextFileType('r'),
        help='text or .gz file containing a list of words to include in class '
             'forming')
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

    if args.method == 'bigram-theano':
        optimizer = TheanoBigramOptimizer(args.num_classes, args.training_set,
                                          args.vocabulary)
    elif args.method == 'bigram-numpy':
        optimizer = NumpyBigramOptimizer(args.num_classes, args.training_set,
                                         args.vocabulary)
    else:
        raise ValueError("Invalid method requested: " + args.method)

    iteration = 1
    while True:
        print("Starting iteration {}.".format(iteration))
        num_words = 0
        num_moves = 0
        for word in optimizer.vocabulary:
            print(word)
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
