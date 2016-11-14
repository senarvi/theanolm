#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from theanolm.iterators.batchiterator import BatchIterator

class LinearBatchIterator(BatchIterator):
    """Iterator for Reading Mini-Batches from a Single File in a Linear Order
    """

    def __init__(self,
                 input_files,
                 vocabulary,
                 batch_size=1,
                 max_sequence_length=None):
        """Constructs an iterator for reading mini-batches from given file or
        memory map.

        :type input_files: file or mmap object, or a list
        :param input_files: input text files or their memory-mapped data

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

        if isinstance(input_files, (list, tuple)):
            self._input_files = input_files
        else:
            self._input_files = [input_files]
        self._reset()

        super().__init__(vocabulary, batch_size, max_sequence_length)

    def _reset(self, shuffle=True):
        """Resets the read pointer back to the beginning of the file.

        :type shuffle: bool
        :param shuffle: also shuffles the input sentences, unless set to False
                        (not supported by this class)
        """

        self._input_file_iter = iter(self._input_files)
        self._input_file = next(self._input_file_iter)
        self._input_file.seek(0)

    def _readline(self):
        """Reads the next input line.

        :rtype: str
        :returns: next line from the data set, or an empty string if the end of
                  the data set is reached.
        """

        result = self._input_file.readline()
        while not result:
            try:
                self._input_file = next(self._input_file_iter)
            except StopIteration:
                return ""
            self._input_file.seek(0)
            result = self._input_file.readline()
        return result
