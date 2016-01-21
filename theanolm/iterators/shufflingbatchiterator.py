#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import sys
import mmap
import logging
import numpy
import theano
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
        """

        self.mmaps = []
        self.pointers = []

        for subset_file in files:
            subset_index = len(self.mmaps)
            subset_mmap = mmap.mmap(subset_file.fileno(),
                                    0,
                                    prot=mmap.PROT_READ)
            self.mmaps.append(subset_mmap)

            print("Finding sentence start positions in {}.".format(
                subset_file.name))
            sys.stdout.flush()
            pointers = [(subset_index, x)
                        for x in find_sentence_starts(subset_mmap)]
            self.pointers.extend(pointers)

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
                 dictionary,
                 batch_size=128,
                 max_sequence_length=100):
        """Initializes the iterator to read sentences in linear order.

        :type input_files: list of file or mmap objects
        :param input_files: input text files

        :type dictionary: Dictionary
        :param dictionary: dictionary that provides mapping between words and
                           word IDs

        :type batch_size: int
        :param batch_size: number of sentences in one mini-batch (unless the end
                           of file is encountered earlier)

        :type max_sequence_length: int
        :param max_sequence_length: if not None, limit to sequences shorter than
                                    this
        """

        self.sentence_pointers = SentencePointers(input_files)
        self.order = list(range(len(self.sentence_pointers)))
        self.order = numpy.asarray(self.order, dtype='int64')
        numpy.random.shuffle(self.order)
        self.next_line = 0

        super().__init__(dictionary, batch_size, max_sequence_length)

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
            h5_iterator['order'][:] = self.order
        else:
            h5_iterator.create_dataset('order', data=self.order)

        h5_iterator.attrs['next_line'] = self.next_line

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
        self.order = h5_iterator['order'].value
        if self.order.size == 0:
            raise IncompatibleStateError("Iteration order is empty in training "
                                         "state.")

        if not 'next_line' in h5_iterator.attrs:
            raise IncompatibleStateError("Current iteration position is "
                                         "missing from training state.")
        self.next_line = int(h5_iterator.attrs['next_line'])
        logging.debug("Restored iterator to line %d of %d.",
                      self.next_line,
                      self.order.size)

    def _reset(self, shuffle=True):
        """Resets the read pointer back to the beginning of the file.
        
        :type shuffle: bool
        :param shuffle: also shuffles the input sentences, unless set to False
        """

        self.next_line = 0
        if shuffle:
            logging.info("Shuffling the order of input lines.")
            numpy.random.shuffle(self.order)

    def _readline(self):
        """Reads the next input line.
        """

        if self.next_line >= self.order.size:
            return ''

        sentence_index = self.order[self.next_line]
        input_file, position = self.sentence_pointers[sentence_index]
        input_file.seek(position)
        line = input_file.readline()
        self.next_line += 1
        return line
