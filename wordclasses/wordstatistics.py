#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy
from scipy.sparse import dok_matrix

class WordStatistics(object):
    """Word Unigram and Bigram Counts
    """

    def __init__(self, corpus_file, vocabulary=None, count_type='int32'):
        """Reads word statistics from corpus file.

        :type corpus_file: file object
        :param corpus_file: a file that contains the input sentences

        :type vocabulary: theanolm.Vocabulary
        :param vocabulary: restrict to these words
        """

        
        vocabulary_size = vocabulary.num_words()
        sos_class_id = vocabulary.word_to_class_id('<s>')
        eos_class_id = vocabulary.word_to_class_id('</s>')
        unk_class_id = vocabulary.word_to_class_id('<unk>')

        self.unigram_counts = numpy.zeros(vocabulary_size, count_type)
        self.bigram_counts = dok_matrix(
            (vocabulary_size, vocabulary_size), dtype=count_type)

        for line in corpus_file:
            sentence = [sos_class_id]
            for word in line.split():
                if word in vocabulary:
                    sentence.append(vocabulary.word_to_id[word])
                else:
                    sentence.append(unk_class_id)
            sentence.append(eos_class_id)
            for word_id in sentence:
                self.unigram_counts[word_id] += 1
            for left_word_id, right_word_id in zip(sentence[:-1], sentence[1:]):
                self.bigram_counts[left_word_id,right_word_id] += 1
