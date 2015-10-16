#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy
import theano

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

class BatchIterator(object):
    """ Iterator for Reading Mini-Batches
    """

    def __init__(self,
                 input_file,
                 dictionary,
                 batch_size=1,
                 max_sequence_length=None):
        """
        :type input_file: file or mmap object
        :param input_file: input text file or its memory-mapped data

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

        self.input_file = input_file
        self.dictionary = dictionary
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.buffer = []
        self.end_of_file = False
        self.input_file.seek(0)

    def __iter__(self):
        return self

    def __next__(self):
        """Returns the next mini-batch read from the file.

        :rtype: tuple of numpy matrices
        :returns: two matrices - one contains the word IDs of each sequence
                  (0 after the last word), and the other contains a mask that
                  is 1 after the last word
        """

        # If EOF was reached on the previous call, but a mini-batch was
        # returned, rewind the file pointer now and raise StopIteration.
        if self.end_of_file:
            self.end_of_file = False
            self._reset()
            raise StopIteration

        sequences = []
        while True:
            sequence = self._read_sequence()
            if sequence is None:
                break
            if len(sequence) < 2:
                continue
            sequences.append(sequence)
            if len(sequences) >= self.batch_size:
                return self._prepare_batch(sequences)

        # When end of file is reached, if no lines were read, rewind to first
        # line and raise StopIteration. If lines were read, return them and
        # raise StopIteration the next time this method is called.
        if len(sequences) == 0:
            self._reset()
            raise StopIteration
        else:
            self.end_of_file = True
            return self._prepare_batch(sequences)

    def __len__(self):
        """Returns the number of mini-batches that the iterator creates at each
        epoch.

        :rtype: int
        :returns: the number of mini-batches that the iterator creates
        """

        self._reset(False)
        num_sequences = 0

        while True:
            sequence = self._read_sequence()
            if sequence is None:
                break
            if len(sequence) < 2:
                continue                
            num_sequences += 1

        self._reset(False)
        return (num_sequences + self.batch_size - 1) // self.batch_size

    def _reset(self, shuffle=True):
        """Resets the read pointer back to the beginning of the file.
        
        :type shuffle: bool
        :param shuffle: also shuffles the input sentences, unless set to False
                        (not supported by this super class)
        """

        self.input_file.seek(0)

    def _read_sequence(self):
        """Returns next word sequence.
        
        If buffer is not empty, returns a sequence from the buffer. Otherwise
        reads a line to the buffer first.
        
        :rtype: list
        :returns: a sequence words (may be empty), or None if no more data
        """

        if len(self.buffer) == 0:
            line = self._readline()
            if len(line) == 0:
                # end of file
                return None
            if type(line) == bytes:
                line = line.decode('utf-8')
            line = line.rstrip()
            if len(line) == 0:
                # empty line
                return []
            self.buffer = line.split()
            self.buffer.append('<sb>')

        if self.max_sequence_length is None:
            result = self.buffer
            self.buffer = []
        else:
            result = self.buffer[:self.max_sequence_length]
            self.buffer = self.buffer[self.max_sequence_length:]
        return result

    def _readline(self):
        """Read the next input line.
        """

        return self.input_file.readline()

    def _prepare_batch(self, sequences):
        """Transposes a list of sequences into a list of time steps. Then
        returns word ID and mask matrices ready to be input to the neural
        network, and a matrix containing the class membership probabilities.

        The first returned matrix contains the word IDs, the second one contains
        the class membership probabilities, and the third one contains a mask
        that defines which elements are past the sequence end - where the other
        matrices contain actual values, the mask matrix contains ones. All the
        elements past the sequence ends will contain zeros.

        All the returned matrices have the same shape. The first dimensions is
        the time step, i.e. the index to a word in a sequence. The second
        dimension selects the sequence. In other words, the first row is the
        first word of each sequence and so on.

        :type sequences: list of lists
        :param sequences: list of sequences, each of which is a list of word
                          IDs

        :rtype: tuple of numpy matrices
        :returns: the word ID, class membership probability, and mask matrix
        """

        num_sequences = len(sequences)
        batch_length = numpy.max([len(s) for s in sequences])

        word_ids = numpy.zeros((batch_length, num_sequences)).astype('int64')
        probs = numpy.zeros((batch_length, num_sequences)).astype(theano.config.floatX)
        mask = numpy.zeros((batch_length, num_sequences)).astype(theano.config.floatX)

        for i, sequence in enumerate(sequences):
            length = len(sequence)
            word_ids[:length, i] = self.dictionary.words_to_ids(sequence)
            probs[:length, i] = self.dictionary.words_to_probs(sequence)
            mask[:length, i] = 1.0

        return word_ids, probs, mask

class ShufflingBatchIterator(BatchIterator):
    """ Iterator for Reading Mini-Batches in a Random Order
    
    Receives the positions of the line starts in the constructor, and shuffles
    the array whenever the end is reached.
    """

    def __init__(self,
                 input_file,
                 dictionary,
                 line_starts,
                 batch_size=128,
                 max_sequence_length=100):
        """
        :type input_file: file object
        :param input_file: input text file

        :type dictionary: Dictionary
        :param dictionary: dictionary that provides mapping between words and
                           word IDs

        :type line_starts: numpy.ndarray
        :param line_starts: a list of start positions of the input sentences;
                            the sentences will be read in the order they appear
                            in this list

        :type batch_size: int
        :param batch_size: number of sentences in one mini-batch (unless the end
                           of file is encountered earlier)

        :type max_sequence_length: int
        :param max_sequence_length: if not None, limit to sequences shorter than
                                    this
        """

        self.line_starts = line_starts
        super().__init__(input_file, dictionary, batch_size, max_sequence_length)
        self.next_line = 0

    def _reset(self, shuffle=True):
        """Resets the read pointer back to the beginning of the file.
        
        :type shuffle: bool
        :param shuffle: also shuffles the input sentences, unless set to False
        """

        self.next_line = 0
        if shuffle:
            logging.info("Shuffling the order of input lines.")
            numpy.random.shuffle(self.line_starts)

    def _readline(self):
        if self.next_line >= len(self.line_starts):
            return ''
        else:
            self.input_file.seek(self.line_starts[self.next_line])
            line = self.input_file.readline()
            self.next_line += 1
            return line
