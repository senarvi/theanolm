#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the WordClass class.
"""

from collections import OrderedDict

import numpy

class WordClass(object):
    """Collection of Words and Their Membership Probabilities

    A word class contains one or more words and their probabilities within
    the class. When a class-based model is not wanted, word classes will be
    created with exactly one word per class.

    The class does not enforce the membership probabilities to sum up to
    one. The user has to call ``normalize_probs()`` after creating the
    class.
    """

    def __init__(self, class_id, word_id, prob):
        """Initializes the class with one word with given probability.

        :type class_id: int
        :param class_id: ID for the class

        :type word_id: int
        :param word_id: ID of the initial word in the class

        :type prob: float
        :param prob: the membership probability of the word
        """

        self.id = class_id
        self._probs = OrderedDict({word_id: prob})

    def add(self, word_id, prob):
        """Adds a word to the class with given probability.

        The membership probabilities are not guaranteed to be normalized.

        :type word_id: int
        :param word_id: ID of the word to add to the class

        :type prob: float
        :param prob: the membership probability of the new word
        """

        self._probs[word_id] = prob

    def get_prob(self, word_id):
        """Returns the class membership probability of a word.

        :type word_id: int
        :param word_id: a word ID that belongs to this class

        :rtype: float
        :returns: the class membership probability of the word
        """

        return self._probs[word_id]

    def set_prob(self, word_id, prob):
        """Sets the class membership probability of a word. The word will be
        added to this class, if it doesn't belong to it already.

        :type word_id: int
        :param word_id: a word ID

        :type prob: float
        :param prob: class membership probability for the word
        """

        self._probs[word_id] = prob

    def normalize_probs(self):
        """Normalizes the class membership probabilities to sum to one.
        """

        prob_sum = sum(self._probs.values())
        for word_id in self._probs:
            self._probs[word_id] /= prob_sum

    def sample(self):
        """Samples a word from the membership probability distribution.

        :rtype: int
        :returns: a random word ID from this class
        """

        word_ids = list(self._probs.keys())
        probs = list(self._probs.values())
        sample_distribution = numpy.random.multinomial(1, probs)
        indices = numpy.flatnonzero(sample_distribution)
        assert len(indices) == 1
        return word_ids[indices[0]]

    def __len__(self):
        """Returns the number of words in this class.

        :rtype: int
        :returns: the number of words in this class
        """

        return len(self._probs)

    def __iter__(self):
        """A generator for iterating through the words in this class.

        :rtype: generator for (int, float)
        :returns: generates a tuple containing a word ID and class
                  membership probability
        """

        for word_id, prob in self._probs.items():
            yield word_id, prob

    def __eq__(self, other):
        """Tests if another word class is exactly the same.

        Two word classes are considered the same if the same word IDs have
        the same probabilities within a tolerance.

        :type other: WordClass
        :param other: another word class

        :rtype: bool
        :returns: True if the classes are the same, False otherwise
        """

        if not isinstance(other, self.__class__):
            return False

        if self.id != other.id:
            return False

        if len(self) != len(other):
            return False

        for word_id, prob in self:
            if not numpy.isclose(prob, other._probs[word_id]):
                return False

        return True

    def __ne__(self, other):
        """Tests if another word class is different.

        Two word classes are considered the same if the same word IDs have
        the same probabilities within a tolerance.

        :type other: WordClass
        :param other: another word class

        :rtype: bool
        :returns: False if the classes are the same, True otherwise
        """

        return not self.__eq__(other)

    def __str__(self):
        """Writes the class members in a string.

        :rtype: str
        :returns: a string showing the class members and their
                  probabilities.
        """

        return '{ ' + \
               ', '.join(str(word_id) + ': ' + str(round(prob, 4))
                         for word_id, prob in self) + \
               ' }'
