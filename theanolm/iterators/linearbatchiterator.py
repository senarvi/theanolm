#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from theanolm.iterators.batchiterator import BatchIterator

class LinearBatchIterator(BatchIterator):
    """Iterator for Reading Mini-Batches from a Single File in a Linear Order
    """

    def __init__(self,
                 input_file,
                 vocabulary,
                 batch_size=1,
                 max_sequence_length=None):
        """Constructs an iterator for reading mini-batches from given file or
        memory map.

        :type input_file: file or mmap object
        :param input_file: input text file or its memory-mapped data

        :type vocabulary: Vocabulary
        :param vocabulary: vocabulary that provides mapping between words and
                           word IDs

        :type batch_size: int
        :param batch_size: number of sentences in one mini-batch (unless the end
                           of file is encountered earlier)

        :type max_sequence_length: int
        :param max_sequence_length: if not None, limit to sequences shorter than
                                    this
        """

        self.input_file = input_file
        self.input_file.seek(0)

        super().__init__(vocabulary, batch_size, max_sequence_length)

    def _reset(self, shuffle=True):
        """Resets the read pointer back to the beginning of the file.
        
        :type shuffle: bool
        :param shuffle: also shuffles the input sentences, unless set to False
                        (not supported by this class)
        """

        self.input_file.seek(0)

    def _readline(self):
        """Reads the next input line.
        """

        return self.input_file.readline()
