#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from theanolm.parsing.batchiterator import BatchIterator

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
            if not self._input_files:
                raise ValueError("LinearBatchIterator constructor expects at "
                                 "least one input file.")
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

        self._file_id = 0
        self._input_file = self._input_files[self._file_id]
        self._input_file.seek(0)

    def _readline(self):
        """Reads the next input line.

        :rtype: tuple of str and int
        :returns: next line from the data set and the index of the file that was
                  used to read it, or None if the end of the data set has been
                  reached.
        """

        line = self._input_file.readline()
        while not line:
            self._file_id += 1
            if self._file_id >= len(self._input_files):
                return None
            self._input_file = self._input_files[self._file_id]
            self._input_file.seek(0)
            line = self._input_file.readline()
        return line, self._file_id
