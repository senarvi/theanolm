#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
from theanolm.parsing.linearbatchiterator import LinearBatchIterator

class ScoringBatchIterator(LinearBatchIterator):
    """Iterator for Reading Mini-Batches for Scoring Text

    Returns the actual words in addition to the word IDs. These are needed for
    subword combination. Only one file can be read.
    """

    def __init__(self, *args, **kwargs):
        """Constructs an iterator for reading mini-batches from given file or
        memory map.
        """

        super().__init__(*args, **kwargs)

    def _prepare_batch(self, sequences):
        """Transposes a list of sequences into a list of time steps. Then
        returns word ID and mask matrices in a format suitable to be input to
        the neural network, and words as a list of lists.

        The first returned matrix contains the word IDs, the second contains the
        word in plane text, and the third contains a mask that defines which
        elements are past the sequence end. Where the other values are valid,
        the mask matrix contains ones.

        The returned matrices have the same shape. The first dimensions is the
        time step, i.e. the index to a word in a sequence. The second dimension
        selects the sequence. In other words, the first row is the first word of
        each sequence and so on. However the plain text words are returned as a
        list of list of strs.

        :type sequences: list of lists
        :param sequences: list of sequences, each of which is a list of word
                          IDs

        :rtype: three ndarrays
        :returns: word ID, word, and mask structures
        """

        num_sequences = len(sequences)
        batch_length = numpy.max([len(s) for s in sequences])

        unk_id = self.vocabulary.word_to_id['<unk>']
        shape = (batch_length, num_sequences)
        word_ids = numpy.ones(shape, numpy.int64) * unk_id
        words = []
        mask = numpy.zeros(shape, numpy.int8)

        for i, sequence in enumerate(sequences):
            length = len(sequence)

            sequence_word_ids = numpy.ones(length, numpy.int64) * unk_id
            sequence_words = []
            for index, (word, file_id) in enumerate(sequence):
                if word in self.vocabulary.word_to_id:
                    sequence_word_ids[index] = self.vocabulary.word_to_id[word]
                sequence_words.append(word)

            word_ids[:length, i] = sequence_word_ids
            words.append(sequence_words)
            mask[:length, i] = 1

        return word_ids, words, mask
