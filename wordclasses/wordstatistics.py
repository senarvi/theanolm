#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
from scipy.sparse import dok_matrix

class WordStatistics(object):
    """Word Unigram and Bigram Counts
    """

    def __init__(self, input_files, vocabulary=None, count_type='int32'):
        """Reads word statistics from corpus file.

        :type input_files: list of file or mmap objects
        :param input_files: input text files

        :type vocabulary: theanolm.Vocabulary
        :param vocabulary: restrict to these words
        """

        vocabulary_size = vocabulary.num_words()
        sos_id = vocabulary.word_to_id['<s>']
        eos_id = vocabulary.word_to_id['</s>']
        unk_id = vocabulary.word_to_id['<unk>']

        self.unigram_counts = numpy.zeros(vocabulary_size, count_type)
        self.bigram_counts = dok_matrix(
            (vocabulary_size, vocabulary_size), dtype=count_type)

        for subset_file in input_files:
            for line in subset_file:
                sentence = [sos_id]
                for word in line.split():
                    if word in vocabulary:
                        sentence.append(vocabulary.word_to_id[word])
                    else:
                        sentence.append(unk_id)
                sentence.append(eos_id)
                for word_id in sentence:
                    self.unigram_counts[word_id] += 1
                for left_word_id, right_word_id in zip(sentence[:-1],
                                                       sentence[1:]):
                    self.bigram_counts[left_word_id,right_word_id] += 1
