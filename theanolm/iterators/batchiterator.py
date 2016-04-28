#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import abstractmethod, ABCMeta
import numpy

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

class BatchIterator(object, metaclass=ABCMeta):
    """Iterator for Reading Mini-Batches
    """

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

        The first returned matrix contains the word IDs, the second identifies
        the file in case of multiple training files, and the third contains a
        mask that defines which elements are past the sequence end. Where the
        other values are valid, the mask matrix contains ones.

        All returned matrices have the same shape. The first dimensions is the
        time step, i.e. the index to a word in a sequence. The second dimension
        selects the sequence. In other words, the first row is the first word of
        each sequence and so on.

        :rtype: tuple of ndarrays
        :returns: word ID and mask matrix
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

    @abstractmethod
    def _reset(self, shuffle=True):
        """Resets the read pointer back to the beginning of the data set.

        :type shuffle: bool
        :param shuffle: also shuffles the input sentences, if supported
        """

        return

    def _read_sequence(self):
        """Returns next word sequence. If sequence length is not limited, it
        will be the next line. Otherwise it may be truncated.

        Start-of-sentence and end-of-sentece tags (``<s>`` and ``</s>``) will be
        inserted at the beginning and the end of the sequence, if they're
        missing. If an empty line is encountered, returns an empty list (instead
        of an empty sentence ``['<s>', '</s>']``).

        If buffer is not empty, returns a sequence from the buffer. Otherwise
        reads a line to the buffer first.

        :rtype: list
        :returns: a sequence of (word, file_id) tuples (may be empty), or None
                  if no more data
        """

        if not self.buffer:
            file_id = self._file_id()
            line = self._readline()
            if len(line) == 0:
                # end of data
                return None
            self.buffer = [(word, file_id)
                           for word in utterance_from_line(line)]

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

    @abstractmethod
    def _readline(self):
        """Reads the next input line.

        :rtype: str
        :returns: next line from the data set, or an empty string if the end of
                  the data set is reached.
        """

        return

    def _file_id(self):
        """When the data set contains multiple files, returns the index of the
        current file. The default implementation returns always 0.

        :rtype: int
        :return: current file index
        """

        return 0

    def _prepare_batch(self, sequences):
        """Transposes a list of sequences into a list of time steps. Then
        returns word ID, file ID, and mask matrices in a format suitable to be
        input to the neural network.

        The first returned matrix contains the word IDs, the second identifies
        the file in case of multiple training files, and the third contains a
        mask that defines which elements are past the sequence end. Where the
        other values are valid, the mask matrix contains ones.

        All returned matrices have the same shape. The first dimensions is the
        time step, i.e. the index to a word in a sequence. The second dimension
        selects the sequence. In other words, the first row is the first word of
        each sequence and so on.

        :type sequences: list of lists
        :param sequences: list of sequences, each of which is a list of word
                          IDs

        :rtype: three ndarrays
        :returns: word ID, file ID, and mask matrix
        """

        num_sequences = len(sequences)
        batch_length = numpy.max([len(s) for s in sequences])

        unk_id = self.vocabulary.word_to_id['<unk>']
        shape = (batch_length, num_sequences)
        word_ids = numpy.ones(shape, numpy.int64) * unk_id
        mask = numpy.zeros(shape, numpy.int8)
        file_ids = numpy.zeros(shape, numpy.int8)

        for i, sequence in enumerate(sequences):
            length = len(sequence)

            sequence_word_ids = numpy.ones(length, numpy.int64) * unk_id
            sequence_file_ids = numpy.zeros(length, numpy.int8)
            for index, (word, file_id) in enumerate(sequence):
                if word in self.vocabulary.word_to_id:
                    sequence_word_ids[index] = self.vocabulary.word_to_id[word]
                sequence_file_ids[index] = file_id

            word_ids[:length, i] = sequence_word_ids
            mask[:length, i] = 1
            file_ids[:length, i] = sequence_file_ids

        return word_ids, file_ids, mask
