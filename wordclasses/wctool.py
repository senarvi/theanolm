#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from theanolm.filetypes import TextFileType
from wordclasses import Optimizer

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
    args = parser.parse_args()

    optimizer = Optimizer(args.num_classes, args.training_set, args.vocabulary)
    iteration = 1
    print("Starting optimization.")
    while True:
        num_moves = optimizer.iterate()
        print("Iteration {}: {} moves, log likelihood {}".format(
            iteration, num_moves, optimizer.log_likelihood()))
        if num_moves == 0:
            break
        iteration += 1

    print("Optimization finished.")
