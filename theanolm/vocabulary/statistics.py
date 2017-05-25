#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module the implements classes and functions for computing statistics from a
corpus.
"""

import numpy
from scipy.sparse import dok_matrix

from theanolm.parsing import utterance_from_line

def compute_word_counts(input_files):
    """Computes word unigram counts using word strings.

    This method does not expect a vocabulary. Start and end of sentence markers
    are not added. Leaves the input files pointing to the beginning of the file.

    :type input_files: list of file or mmap objects
    :param input_files: input text files

    :rtype: dict
    :returns: a mapping from word strings to counts
    """

    result = dict()
    for subset_file in input_files:
        for line in subset_file:
            for word in utterance_from_line(line):
                if word not in result:
                    result[word] = 1
                else:
                    result[word] += 1
        subset_file.seek(0)
    return result

class BigramStatistics(object):
    """Word Unigram and Bigram Counts
    """

    def __init__(self, input_files, vocabulary=None, count_type='int32'):
        """Reads word statistics from corpus file and creates the
        ``unigram_counts`` and ``bigram_counts`` attributes.

        Leaves the input files pointing to the beginning of the file.

        :type input_files: list of file or mmap objects
        :param input_files: input text files

        :type vocabulary: Vocabulary
        :param vocabulary: restrict to these words
        """

        vocabulary_size = vocabulary.num_words()
        unk_id = vocabulary.word_to_id['<unk>']

        self.unigram_counts = numpy.zeros(vocabulary_size, count_type)
        self.bigram_counts = dok_matrix(
            (vocabulary_size, vocabulary_size), dtype=count_type)

        for subset_file in input_files:
            for line in subset_file:
                sequence = []
                for word in utterance_from_line(line):
                    if word in vocabulary:
                        sequence.append(vocabulary.word_to_id[word])
                    else:
                        sequence.append(unk_id)
                for word_id in sequence:
                    self.unigram_counts[word_id] += 1
                for left_word_id, right_word_id in zip(sequence[:-1],
                                                       sequence[1:]):
                    self.bigram_counts[left_word_id,right_word_id] += 1
            subset_file.seek(0)
