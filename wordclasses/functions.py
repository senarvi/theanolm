#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def byte_size(x):
    """Converts a byte size into a human-readable string.

    :type x: int
    :param x: number of bytes

    :rtype: str
    :returns: the given size formatted in a human-readable string
    """

    suffixes = ['bytes', 'KB', 'MB', 'GB', 'TB']
    index = 0
    while x > 1024 and index < 4:
        index += 1
        x /= 1024
    return "{} {}".format(int(round(x)), suffixes[index])

def is_scheduled(num_words, frequency, words_per_iteration):
    """Checks if an event is scheduled to be performed within given number
    of updates after this point.

    For example, word_per_iteration=9, frequency=2:

    num_words:      1   2   3   4  [5]  6   7   8  [9] 10  11  12
    * frequency:    2   4   6   8  10  12  14  16  18  20  22  24
    modulo:         2   4   6   8   1   3   5   7   0   2   4   6

    :type num_words: int
    :param num_words: number of words so far

    :type frequency: int
    :param frequency: how many times per iteration the event should be performed

    :type words_per_iteration: int
    :param words_per_iteration: number of words in one iteration

    :rtype: bool
    :returns: whether the operation is scheduled to be performed
    """

    modulo = num_words * frequency % words_per_iteration
    return modulo < frequency
