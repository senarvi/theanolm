#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import mmap
import logging
import numpy
from numpy import random
from theanolm.iterators.batchiterator import BatchIterator

def find_sentence_starts(data):
    """Finds the positions inside a memory-mapped file, where the sentences
    (lines) start.

    TextIOWrapper disables tell() when readline() is called, so search for
    sentence starts in memory-mapped data.

    :type data: mmap.mmap
    :param data: memory-mapped data of the input file

    :rtype: list of ints
    :returns: a list of file offsets pointing to the next character from a
              newline (including file start and excluding file end)
    """

    result = [0]

    pos = 0
    while True:
        pos = data.find(b'\n', pos)
        if pos == -1:
            break;
        pos += 1
        if pos < len(data):
            result.append(pos)

    return result

class SentencePointers(object):
    """A class that creates a memory map of text files and stores pointers to
    the beginning of each line in each file.
    """

    def __init__(self, files):
        """Creates a memory map of the given files and finds the sentence
        starts.

        The pointers to sentence starts will be saved in a structure where each
        element is a tuple of two indices - the first index will select the file
        from the mmaps list and the second index points to the position inside
        the file.

        Also saves in ``pointer_ranges`` an index to the first pointer and one
        past the last pointer of each file.

        :type files: list of file objects
        :param files: input text files
        """

        self.mmaps = []
        self.pointers = []
        self.pointer_ranges = []

        for subset_file in files:
            subset_index = len(self.mmaps)
            subset_mmap = mmap.mmap(subset_file.fileno(),
                                    0,
                                    prot=mmap.PROT_READ)
            self.mmaps.append(subset_mmap)

            logging.debug("Finding sentence start positions in %s.",
                          subset_file.name)
            sys.stdout.flush()
            pointers = [(subset_index, x)
                        for x in find_sentence_starts(subset_mmap)]
            pointers_start = len(self.pointers)
            self.pointers.extend(pointers)
            pointers_stop = len(self.pointers)
            self.pointer_ranges.append((pointers_start, pointers_stop))

    def __len__(self):
        """Returns the number of sentences.

        :rtype: int
        :returns: the number of sentences found
        """

        return len(self.pointers)

    def __getitem__(self, sentence_index):
        """Returns a pointer to sentence with given index.
        
        :type sentence_index: int
        :param sentence_index: a linear index between zero and one less the
                               total number of sentences

        :rtype: tuple of a file object and int
        :returns: a file object and a pointer to the file
        """

        subset_index, sentence_start = self.pointers[sentence_index]
        subset_mmap = self.mmaps[subset_index]
        return (subset_mmap, sentence_start)

class ShufflingBatchIterator(BatchIterator):
    """Iterator for Reading Mini-Batches in a Random Order
    
    Receives the positions of the line starts in the constructor, and shuffles
    the array whenever the end is reached.
    """

    def __init__(self,
                 input_files,
                 sampling,
                 vocabulary,
                 batch_size=128,
                 max_sequence_length=100):
        """Initializes the iterator to read sentences in linear order.

        :type input_files: list of file objects
        :param input_files: input text files

        :type sampling: list of floats
        :param sampling: specifies a fraction for each input file, how much to
                         sample on each epoch

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

        self._sentence_pointers = SentencePointers(input_files)

        self._sample_sizes = []
        fraction_iter = iter(sampling)
        for (start, stop) in self._sentence_pointers.pointer_ranges:
            fraction = next(fraction_iter, 1.0)
            sample_size = round(fraction * (stop - start))
            self._sample_sizes.append(sample_size)

        self._next_line = 0
        self._order = numpy.arange(sum(self._sample_sizes), dtype='int64')
        self._reset()

        super().__init__(vocabulary, batch_size, max_sequence_length)

    def get_state(self, state):
        """Saves the iterator state in a HDF5 file.

        Sets ``iterator/order`` to the iteration order, and
        ``iterator/next_line`` the index to the next sentence in the list. Note
        that if the program is restarted, the same training files have to be
        loaded in order for this to work. If there already is a
        ``iterator/order`` in the state, it will be replaced, so it has to have
        the same number of elements.

        :type state: h5py.File
        :param state: HDF5 file for storing the iterator state
        """

        h5_iterator = state.require_group('iterator')

        if 'order' in h5_iterator:
            h5_iterator['order'][:] = self._order
        else:
            h5_iterator.create_dataset('order', data=self._order)

        h5_iterator.attrs['next_line'] = self._next_line

    def set_state(self, state):
        """Restores the iterator state.

        Sets the offsets to the sentence starts (the order in which they are
        iterated), and the index to the current sentence.
        
        Requires that ``state`` contains values for all the iterator parameters.

        :type state: h5py.File
        :param state: HDF5 file that contains the iterator state
        """

        if not 'iterator' in state:
            raise IncompatibleStateError("Iterator state is missing.")
        h5_iterator = state['iterator']

        if not 'order' in h5_iterator:
            raise IncompatibleStateError("Iteration order is missing from "
                                         "training state.")
        self._order = h5_iterator['order'].value
        if self._order.size == 0:
            raise IncompatibleStateError("Iteration order is empty in training "
                                         "state.")

        if not 'next_line' in h5_iterator.attrs:
            raise IncompatibleStateError("Current iteration position is "
                                         "missing from training state.")
        self._next_line = int(h5_iterator.attrs['next_line'])
        logging.debug("Restored iterator to line %d of %d.",
                      self._next_line,
                      self._order.size)

    def _create_order(self):
        """Creates a random iteration order.
        """

    def _reset(self, shuffle=True):
        """Resets the read pointer back to the beginning of the data set. If
        ``shuffle`` is set to True, also creates a new random order for
        iterating the input lines.

        :type shuffle: bool
        :param shuffle: also shuffles the input sentences, unless set to False
        """

        self._next_line = 0
        if shuffle:
            logging.debug("Generating a random order of input lines.")

            samples = []
            for (start, stop), sample_size in \
                zip(self._sentence_pointers.pointer_ranges, self._sample_sizes):

                population = numpy.arange(start, stop, dtype='int64')
                # No duplicates, unless we need more sentences than there are
                # in the file.
                replace = sample_size > len(population)
                sample = random.choice(population, sample_size, replace=replace)
                samples.append(sample)
            self._order = numpy.concatenate(samples)
            for _ in range(10):
                random.shuffle(self._order)

    def _readline(self):
        """Reads the next input line.

        :rtype: str
        :returns: next line from the data set, or an empty string if the end of
                  the data set is reached.
        """

        if self._next_line >= self._order.size:
            return ''

        sentence_index = self._order[self._next_line]
        input_file, position = self._sentence_pointers[sentence_index]
        input_file.seek(position)
        line = input_file.readline()
        self._next_line += 1
        return line

    def _file_id(self):
        """When the data set contains multiple files, returns the index of the
        current file.

        :rtype: int
        :return: current file index
        """

        if self._next_line >= self._order.size:
            return 0

        sentence_index = self._order[self._next_line]
        subset_index, _ = self._sentence_pointers.pointers[sentence_index]
        return subset_index
