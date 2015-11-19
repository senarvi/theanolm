#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
from theanolm.exceptions import InputError

class Dictionary(object):
    """Word Dictionary

    Dictionary class provides a mapping between the words and word or class IDs.
    """

    class WordClass(object):
        """Collection of Words and Their Membership Probabilities

        A word class contains one or more words and their probabilities within
        the class. When a class-based model is not wanted, word classes will be
        created with exactly one word per class.

        The class does not enforce the membership probabilities to sum up to
        one. The user has to call ``normalize_probs()`` after creating the
        class.
        """

        def __init__(self, word, prob):
            """Initializes the class with one word with given probability.

            :type word: string
            :param word: the initial word in the class

            :type prob: float
            :param prob: the membership probability of the word
            """

            self.id = None
            self._probs = OrderedDict({word: prob})

        def add(self, word, prob):
            """Adds a word to the class with given probability.

            The membership probabilities are not guaranteed to be normalized.

            :type word: string
            :param word: the word to add to the class

            :type prob: float
            :param prob: the membership probability of the new word
            """

            self._probs[word] = prob

        def get_prob(self, word):
            """Returns the class membership probability of a word.

            :type word: string
            :param word: a word that belongs to this class

            :rtype: float
            :returns: the class membership probability of the word
            """

            return self._probs[word]

        def normalize_probs(self):
            """Normalizes the class membership probabilities to sum to one.
            """

            prob_sum = sum(self._probs.values())
            for word in self._probs:
                self._probs[word] /= prob_sum

        def sample(self):
            """Samples a word from the membership probability distribution.

            :rtype: str
            :returns: a random word from this class
            """

            words = list(self._probs.keys())
            probs = list(self._probs.values())
            sample_distribution = numpy.random.multinomial(1, probs)
            indices = numpy.flatnonzero(sample_distribution)
            assert len(indices) == 1
            return words[indices[0]]

    def __init__(self, input_file, input_format):
        """Creates word classes.

        If ``input_format`` is one of:
        * "words": ``input_file`` contains one word per line. Each word will be
                   assigned to its own class.
        * "classes": ``input_file`` contains a word followed by whitespace
                     followed by class ID on each line. Each word will be
                     assigned to the specified class. The class IDs can be
                     anything; they will be translated to consecutive numbers
                     after reading the file.
        * "srilm-classes": ``input_file`` contains a class name, membership
                           probability, and word, separated by whitespace, on
                           each line.

        :type input_file: file object
        :param input_file: input dictionary file

        :type input_format str
        :param input_format: format of the input dictionary file, "words",
	                     "classes", or "srilm-classes"
        """

        # The word classes with consecutive indices. The first three classes are
        # the start-of-sentence, end-of-sentence, and unknown word tokens.
        self._word_classes = [Dictionary.WordClass('<s>', 1.0),
                              Dictionary.WordClass('</s>', 1.0),
                              Dictionary.WordClass('<UNK>', 1.0)]
        # Mapping from the IDs in the file to our word classes.
        file_id_to_class = dict()
        # Mapping from word strings to word classes.
        self._word_to_class = {'<s>': self._word_classes[0],
                               '</s>': self._word_classes[1],
                               '<UNK>': self._word_classes[2]}
        self.sos_id = 0
        self.eos_id = 1
        self.unk_id = 2

        for line in input_file:
            line = line.strip()
            fields = line.split()
            if len(fields) == 0:
                continue
            if input_format == 'words' and len(fields) == 1:
                word = fields[0]
                file_id = None
                prob = 1.0
            elif input_format == 'classes' and len(fields) == 2:
                word = fields[0]
                file_id = int(fields[1])
                prob = 1.0
            elif input_format == 'srilm-classes' and len(fields) == 3:
                file_id = fields[0]
                prob = float(fields[1])
                word = fields[2]
            else:
                raise InputError("%d fields on one line of dictionary file: %s" % (len(fields), line))

            if word in self._word_to_class:
                raise InputError("Word `%s' appears more than once in the dictionary file." % word)
            if file_id in file_id_to_class:
                word_class = file_id_to_class[file_id]
                word_class.add(word, prob)
            else:
                # No ID in the file or a new ID.
                word_class = Dictionary.WordClass(word, prob)
                self._word_classes.append(word_class)
                if not file_id is None:
                    file_id_to_class[file_id] = word_class
            self._word_to_class[word] = word_class

        for i, word_class in enumerate(self._word_classes):
            word_class.id = i
            word_class.normalize_probs()

    def num_words(self):
        """Returns the number of words in the dictionary.

        :rtype: int
        :returns: the number of words in the dictionary
        """

        return len(self._word_to_class)

    def num_classes(self):
        """Returns the number of word classes.

        :rtype: int
        :returns: the number of words classes
        """

        return len(self._word_classes)

    def words_to_ids(self, words):
        """Translates words into word (class) IDs.

        :type words: list of strings
        :param words: a list of words

        :rtype: list of ints
        :returns: the given words translated into word IDs
        """

        return [self._word_to_class[word].id
                if word in self._word_to_class
                else self.unk_id
                for word in words]

    def ids_to_words(self, word_ids):
        """Translates word (class) IDs into words. If classes are used, samples
        a word from the membership probability distribution.

        :type word_ids: list of ints
        :param word_ids: a list of word IDs

        :rtype: list of strings
        :returns: the given word IDs translated into words
        """

        return [self._word_classes[word_id].sample()
                for word_id in word_ids]

    def words_to_probs(self, words):
        """Returns a list of class membership probabilities for each input word.

        :type words: numpy.ndarray of strs
        :param words: a vector or matrix of words

        :rtype: numpy.ndarray of floats
        :returns: a matrix the same shape as ``words`` containing the class
                  membership probabilities
        """

        # <unk> has class membership probability 1.0.
        return [self._word_to_class[word].get_prob(word)
                if word in self._word_to_class
                else 1.0
                for word in words]
