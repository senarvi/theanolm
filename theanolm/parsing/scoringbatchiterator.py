#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements an iterator for reading mini-batches for scoring.
"""

import numpy

from theanolm.parsing.linearbatchiterator import LinearBatchIterator

class ScoringBatchIterator(LinearBatchIterator):
    """Iterator for Reading Mini-Batches for Scoring Text

    Returns the actual words in addition to the word IDs. These are needed for
    subword combination. File IDs are not returned.
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

        The first returned matrix contains the word IDs. Word other than the
        shortlist words may be mapped to <unk>, depending on
        ``self._map_oos_to_unk``. The second matrix contains the words in plain
        text, and the third one contains a mask that defines which elements are
        past the sequence end. Where the other values are valid, the mask matrix
        contains ones.

        The word ID and mask matrices have the same shape. The first dimensions
        is the time step, i.e. the index to a word in a sequence. The second
        dimension selects the sequence. In other words, the first row is the
        first word of each sequence and so on. The plain text words are returned
        as a list of sequences, each sequence extending only to the sequence
        end.

        :type sequences: list of lists
        :param sequences: list of sequences, each of which is a list of word
                          IDs

        :rtype: three ndarrays
        :returns: word ID, word, and mask structures
        """

        num_sequences = len(sequences)
        batch_length = numpy.max([len(s) for s in sequences])

        unk_id = self._vocabulary.word_to_id['<unk>']
        shape = (batch_length, num_sequences)
        word_ids = numpy.ones(shape, numpy.int64) * unk_id
        words = []
        mask = numpy.zeros(shape, numpy.int8)

        for i, sequence in enumerate(sequences):
            length = len(sequence)

            sequence_word_ids = numpy.ones(length, numpy.int64) * unk_id
            sequence_words = []
            for index, (word, file_id) in enumerate(sequence):
                if word in self._vocabulary:
                    word_id = self._vocabulary.word_to_id[word]
                    if (not self._map_oos_to_unk) or \
                       self._vocabulary.in_shortlist(word_id):
                        sequence_word_ids[index] = word_id
                sequence_words.append(word)

            word_ids[:length, i] = sequence_word_ids
            words.append(sequence_words)
            mask[:length, i] = 1

        return word_ids, words, mask
