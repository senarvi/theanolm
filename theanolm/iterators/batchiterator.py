#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy
import theano

def utterance_from_line(line):
    """Converts a line of text, read from an input file, into a list of words.

    Start-of-sentence and end-of-sentece tags (``<s>`` and ``</s>``) will be
    inserted at the beginning and the end of the list, if they're missing. If
    the line is empty, returns an empty list (instead of an empty sentence
    ``['<s>', '</s>']``).

    :type line: str or bytes
    :param line: a line of text (read from an input file)
    """

    if type(line) == bytes:
        line = line.decode('utf-8')
    line = line.rstrip()
    if len(line) == 0:
        # empty line
        return []

    result = line.split()
    if result[0] != '<s>':
        result.insert(0, '<s>')
    if result[-1] != '</s>':
        result.append('</s>')

    return result

class BatchIterator(object):
    """Iterator for Reading Mini-Batches
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 vocabulary,
                 batch_size=1,
                 max_sequence_length=None):
        """Constructs an iterator for reading mini-batches from given file or
        memory map.

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

        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.buffer = []
        self.end_of_file = False

    def __iter__(self):
        return self

    def __next__(self):
        """Returns the next mini-batch read from the file.

        :rtype: tuple of numpy matrices
        :returns: the word ID, class membership probability, and mask matrix
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
        if not sequences:
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

    @abc.abstractmethod
    def _reset(self, shuffle=True):
        """Resets the read pointer back to the beginning of the file.

        :type shuffle: bool
        :param shuffle: also shuffles the input sentences, if supported
        """

        return

    def _read_sequence(self):
        """Returns next word sequence.

        Start-of-sentence and end-of-sentece tags (``<s>`` and ``</s>``) will be
        inserted at the beginning and the end of the sequence, if they're
        missing. If an empty line is encountered, returns an empty list (instead
        of an empty sentence ``['<s>', '</s>']``).

        If buffer is not empty, returns a sequence from the buffer. Otherwise
        reads a line to the buffer first.

        :rtype: list
        :returns: a sequence of words (may be empty), or None if no more data
        """

        if not self.buffer:
            line = self._readline()
            if len(line) == 0:
                # end of file
                return None
            self.buffer = utterance_from_line(line)

        if self.max_sequence_length is None:
            result = self.buffer
            self.buffer = []
        else:
            result = self.buffer[:self.max_sequence_length]
            # XXX Disable splitting of sentences into several sequences. It
            # might have a negative effect since what is assumed to be before
            # the continuation sequence is a zero vector. XXX
            #self.buffer = self.buffer[self.max_sequence_length:]
            self.buffer = []
        return result

    @abc.abstractmethod
    def _readline(self):
        """Reads the next input line.
        """

        return

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
        :returns: class ID, class membership probability, and mask matrix
        """

        num_sequences = len(sequences)
        batch_length = numpy.max([len(s) for s in sequences])

        class_ids = numpy.zeros((batch_length, num_sequences), numpy.int64)
        probs = numpy.zeros((batch_length, num_sequences)).astype(theano.config.floatX)
        mask = numpy.zeros((batch_length, num_sequences), numpy.int8)

        for i, sequence in enumerate(sequences):
            length = len(sequence)
            class_ids[:length, i] = self.vocabulary.words_to_class_ids(sequence)
            probs[:length, i] = self.vocabulary.words_to_probs(sequence)
            mask[:length, i] = 1

        return class_ids, probs, mask
