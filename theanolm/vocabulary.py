#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
from theanolm.exceptions import InputError

class Vocabulary(object):
    """Word or Class Vocabulary

    Vocabulary class provides a mapping between the words and word or class IDs.
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

        def __init__(self, id, word, prob):
            """Initializes the class with one word with given probability.

            :type id: int
            :param id: ID for the class

            :type word: str
            :param word: the initial word in the class

            :type prob: float
            :param prob: the membership probability of the word
            """

            self.id = id
            self._probs = OrderedDict({word: prob})

        def add(self, word, prob):
            """Adds a word to the class with given probability.

            The membership probabilities are not guaranteed to be normalized.

            :type word: str
            :param word: the word to add to the class

            :type prob: float
            :param prob: the membership probability of the new word
            """

            self._probs[word] = prob

        def get_prob(self, word):
            """Returns the class membership probability of a word.

            :type word: str
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

        def name(self):
            """If the class contains only one word, returns the word. Otherwise
            returns CLASS-12345, where 12345 is the internal class ID.

            :type word_ids: list of ints
            :param word_ids: a list of word IDs

            :rtype: list of strings
            :returns: the given word IDs translated into words
            """

            if len(self._probs) == 1:
                return next(iter(self._probs))
            elif self.id is None:
                return 'CLASS'
            else:
                return 'CLASS-{0:05d}'.format(self.id)

    def __init__(self, word_classes, word_to_class_id):
        """Creates a vocabulary based on given word-to-class mapping.

        :type word_classes: list of WordClass objects
        :param word_classes: list of all the word classes

        :type word_to_class_id: dict
        :param word_to_class_id: mapping from words to indices in
                                 ``word_classes``
        """

        self.word_to_id = dict()
        self._word_classes = []
        self.word_id_to_class_id = []

        if not '<s>' in word_to_class_id:
            word_id = len(self.word_id_to_class_id)
            class_id = len(self._word_classes)
            self.word_to_id['<s>'] = word_id
            self.word_id_to_class_id.append(class_id)
            word_class = Vocabulary.WordClass(class_id, '<s>', 1.0)
            self._word_classes.append(word_class)

        if not '</s>' in word_to_class_id:
            word_id = len(self.word_id_to_class_id)
            class_id = len(self._word_classes)
            self.word_to_id['</s>'] = word_id
            self.word_id_to_class_id.append(class_id)
            word_class = Vocabulary.WordClass(class_id, '</s>', 1.0)
            self._word_classes.append(word_class)

        if not '<unk>' in word_to_class_id:
            word_id = len(self.word_id_to_class_id)
            class_id = len(self._word_classes)
            self.word_to_id['<unk>'] = word_id
            self.word_id_to_class_id.append(class_id)
            word_class = Vocabulary.WordClass(class_id, '<unk>', 1.0)
            self._word_classes.append(word_class)

        self.first_normal_word_id = len(self.word_id_to_class_id)
        self.first_normal_class_id = len(self._word_classes)

        for word, class_id in word_to_class_id.items():
            word_id = len(self.word_id_to_class_id)
            self.word_to_id[word] = word_id
            self.word_id_to_class_id.append(self.first_normal_class_id + class_id)

        self._word_classes.extend(word_classes)
        for word_class in self._word_classes:
            word_class.normalize_probs()

    @classmethod
    def from_file(classname, input_file, input_format):
        """Reads vocabulary and possibly word classes from a text file.

        ``input_format`` is one of:
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
        :param input_file: input vocabulary file

        :type input_format str
        :param input_format: format of the input vocabulary file, "words",
	                     "classes", or "srilm-classes"
        """

        word_classes = []
        word_to_class_id = dict()
        # Mapping from the IDs in the file to our internal class IDs.
        file_id_to_class_id = dict()

        for line in input_file:
            line = line.strip()
            fields = line.split()
            if not fields:
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
                raise InputError("%d fields on one line of vocabulary file: %s" % (len(fields), line))

            if word in word_to_class_id:
                raise InputError("Word `%s' appears more than once in the vocabulary file." % word)
            if file_id in file_id_to_class_id:
                class_id = file_id_to_class_id[file_id]
                word_classes[class_id].add(word, prob)
            else:
                # No ID in the file or a new ID.
                class_id = len(word_classes)
                word_class = Vocabulary.WordClass(class_id, word, prob)
                word_classes.append(word_class)
                if not file_id is None:
                    file_id_to_class_id[file_id] = class_id
            word_to_class_id[word] = class_id

        return classname(word_classes, word_to_class_id)

    @classmethod
    def from_word_counts(classname, word_counts, num_classes=None):
        """Creates a vocabulary and dummy classes from word counts.

        :type word_counts: dict
        :param word_counts: dictionary from words to the number of occurrences
                            in the corpus

        :type num_classes: int
        :param num_classes: number of classes to create in addition to the
                            special classes, or None for one class per word
        """

        word_classes = []
        word_to_class_id = dict()

        if num_classes is None:
            num_classes = len(word_counts)

        class_id = 0
        for word, _ in sorted(word_counts.items(),
                              key=lambda x: x[1]):
            if class_id < len(word_classes):
                word_classes[class_id].add(word, 1.0)
            else:
                assert class_id == len(word_classes)
                word_class = Vocabulary.WordClass(class_id, word, 1.0)
                word_classes.append(word_class)
            word_to_class_id[word] = class_id
            class_id = (class_id + 1) % num_classes

        return classname(word_classes, word_to_class_id)

    @classmethod
    def from_corpus(classname, input_files, num_classes=None):
        """Creates a vocabulary based on word counts from training set.

        :type input_files: list of file or mmap objects
        :param input_files: input text files

        :type num_classes: int
        :param num_classes: number of classes to create in addition to the
                            special classes, or None for one class per word
        """

        word_counts = dict()

        for subset_file in input_files:
            for line in subset_file:
                for word in line.split():
                    if not word in word_counts:
                        word_counts[word] = 1
                    else:
                        word_counts[word] += 1

        return classname.from_word_counts(word_counts, num_classes)

    def num_words(self):
        """Returns the number of words in the vocabulary.

        :rtype: int
        :returns: the number of words in the vocabulary
        """

        return len(self.word_id_to_class_id)

    def num_classes(self):
        """Returns the number of word classes.

        :rtype: int
        :returns: the number of words classes
        """

        return len(self._word_classes)

    def word_to_class_id(self, word):
        """Returns the class ID of given word.

        :type word: str
        :param word: a word

        :rtype: int
        :returns: ID of the class where ``word`` is assigned to
        """

        return self.word_id_to_class_id[self.word_to_id[word]]

    def words_to_class_ids(self, words):
        """Translates words into class IDs.

        :type words: list of strs
        :param words: a list of words

        :rtype: list of ints
        :returns: the given words translated into class IDs
        """

        return [ self.word_to_class_id(word)
                 if word in self.word_to_id
                 else self.unk_id
                 for word in words ]

    def ids_to_words(self, word_ids):
        """Translates class IDs into words. If classes are used, samples
        a word from the membership probability distribution.

        :type word_ids: list of ints
        :param word_ids: a list of word IDs

        :rtype: list of strings
        :returns: the given word IDs translated into words
        """

        return [self._word_classes[word_id].sample()
                for word_id in word_ids]

    def ids_to_names(self, word_ids):
        """Translates word / class IDs into word / class names. If a class
        contains only one word, class name will be the word. Otherwise class
        name will be CLASS-12345, where 12345 is the internal class ID.

        :type word_ids: list of ints
        :param word_ids: a list of word IDs

        :rtype: list of strings
        :returns: class names of the given word IDs
        """

        return [self._word_classes[word_id].name()
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
        return [ self._word_classes[self.word_to_class_id(word)].get_prob(word)
                 if word in self.word_to_id
                 else 1.0
                 for word in words ]

    def __contains__(self, word):
        """Tests if ``word`` is included in the vocabulary.

        :type word: str
        :param word: a word

        :rtype: bool
        :returns: True if ``word`` is in the vocabulary, False otherwise.
        """

        return word in self.word_to_id

    def words(self):
        """A generator for all the words in the vocabulary.

        :rtype: generator of str
        :returns: iterates through the words
        """

        for word in self.word_to_id.keys():
            yield word
