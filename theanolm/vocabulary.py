#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import h5py
from theanolm.exceptions import IncompatibleStateError, InputError

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

    def __init__(self, id_to_word, word_id_to_class_id, word_classes):
        """If the special tokens <s>, </s>, and <unk> don't exist in the word
        list, adds them and creates a separate class for each token. Then
        constructs a vocabulary based on given word-to-class mapping.

        :type id_to_word: list of strs
        :param id_to_word: mapping from word IDs to word names

        :type word_id_to_class_id: list of ints
        :param word_id_to_class_id: mapping from word IDs to indices in
                                    ``word_classes``

        :type word_classes: list of WordClass objects
        :param word_classes: list of all the word classes
        """

        if not '<s>' in id_to_word:
            word_id = len(id_to_word)
            assert word_id == len(word_id_to_class_id)
            class_id = len(word_classes)
            id_to_word.append('<s>')
            word_id_to_class_id.append(class_id)
            word_class = Vocabulary.WordClass(class_id, word_id, 1.0)
            word_classes.append(word_class)

        if not '</s>' in id_to_word:
            word_id = len(id_to_word)
            assert word_id == len(word_id_to_class_id)
            class_id = len(word_classes)
            id_to_word.append('</s>')
            word_id_to_class_id.append(class_id)
            word_class = Vocabulary.WordClass(class_id, word_id, 1.0)
            word_classes.append(word_class)

        if not '<unk>' in id_to_word:
            word_id = len(id_to_word)
            assert word_id == len(word_id_to_class_id)
            class_id = len(word_classes)
            id_to_word.append('<unk>')
            word_id_to_class_id.append(class_id)
            word_class = Vocabulary.WordClass(class_id, word_id, 1.0)
            word_classes.append(word_class)

        index = len(word_classes) - 1
        while True:
            word_class = word_classes[index]
            if len(word_class) == 1:
                word_id, _ = next(iter(word_class))
                if id_to_word[word_id].startswith('<'):
                    index -= 1
                    continue
            break
        self.num_normal_classes = index + 1

        for word_class in word_classes:
            word_class.normalize_probs()

        self.id_to_word = numpy.asarray(id_to_word, dtype=object)
        self.word_id_to_class_id = numpy.asarray(word_id_to_class_id)
        self._word_classes = numpy.asarray(word_classes)
        self.word_to_id = {word: word_id
                           for word_id, word in enumerate(self.id_to_word)}

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

        :type input_format: str
        :param input_format: format of the input vocabulary file, "words",
                             "classes", or "srilm-classes"
        """

        # We have also a set of the words just for faster checking if a word has
        # already been encountered.
        words = set()
        id_to_word = []
        word_id_to_class_id = []
        word_classes = []
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

            if word in words:
                raise InputError("Word `%s' appears more than once in the vocabulary file." % word)
            words.add(word)
            word_id = len(id_to_word)
            id_to_word.append(word)

            if file_id in file_id_to_class_id:
                class_id = file_id_to_class_id[file_id]
                word_classes[class_id].add(word_id, prob)
            else:
                # No ID in the file or a new ID.
                class_id = len(word_classes)
                word_class = Vocabulary.WordClass(class_id, word_id, prob)
                word_classes.append(word_class)
                if not file_id is None:
                    file_id_to_class_id[file_id] = class_id

            assert word_id == len(word_id_to_class_id)
            word_id_to_class_id.append(class_id)

        return classname(id_to_word, word_id_to_class_id, word_classes)

    @classmethod
    def from_word_counts(classname, word_counts, num_classes=None):
        """Creates a vocabulary and classes from word counts.

        :type word_counts: dict
        :param word_counts: dictionary from words to the number of occurrences
                            in the corpus

        :type num_classes: int
        :param num_classes: number of classes to create in addition to the
                            special classes, or None for one class per word
        """

        id_to_word = []
        word_id_to_class_id = []
        word_classes = []

        if num_classes is None:
            num_classes = len(word_counts)

        class_id = 0
        for word, _ in sorted(word_counts.items(),
                              key=lambda x: x[1]):
            word_id = len(id_to_word)
            id_to_word.append(word)

            if class_id < len(word_classes):
                word_classes[class_id].add(word_id, 1.0)
            else:
                assert class_id == len(word_classes)
                word_class = Vocabulary.WordClass(class_id, word_id, 1.0)
                word_classes.append(word_class)

            assert word_id == len(word_id_to_class_id)
            word_id_to_class_id.append(class_id)
            class_id = (class_id + 1) % num_classes

        return classname(id_to_word, word_id_to_class_id, word_classes)

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

    @classmethod
    def from_state(classname, state):
        """Reads the vocabulary from a network state.

        :type state: hdf5.File
        :param state: HDF5 file that contains the architecture parameters
        """

        if not 'vocabulary' in state:
            raise IncompatibleStateError(
                "Vocabulary is missing from neural network state.")
        h5_vocabulary = state['vocabulary']

        if not 'words' in h5_vocabulary:
            raise IncompatibleStateError(
                "Vocabulary parameter 'words' is missing from neural network "
                "state.")
        id_to_word = h5_vocabulary['words'].value

        if not 'classes' in h5_vocabulary:
            raise IncompatibleStateError(
                "Vocabulary parameter 'classes' is missing from neural network "
                "state.")
        word_id_to_class_id = h5_vocabulary['classes'].value

        if not 'probs' in h5_vocabulary:
            raise IncompatibleStateError(
                "Vocabulary parameter 'probs' is missing from neural network "
                "state.")
        num_classes = word_id_to_class_id.max() + 1
        word_classes = [None] * num_classes
        h5_probs = h5_vocabulary['probs'].value
        for word_id, prob in enumerate(h5_probs):
            class_id = word_id_to_class_id[word_id]
            if word_classes[class_id] is None:
                word_class = Vocabulary.WordClass(class_id, word_id, prob)
                word_classes[class_id] = word_class
            else:
                word_classes[class_id].add(word_id, prob)

        return classname(id_to_word.tolist(),
                         word_id_to_class_id.tolist(),
                         word_classes)

    def compute_probs(self, input_files):
        """Recomputes unigram class membership probabilities from text files.
        Probabilities are updates only for classes whose words occur in the
        text.

        :type input_files: list of file or mmap objects
        :param input_files: input text files
        """

        counts = numpy.zeros(self.num_words(), dtype='int64')
        for subset_file in input_files:
            for line in subset_file:
                for word in line.split():
                    if word in self.word_to_id:
                        counts[self.word_to_id[word]] += 1

        for cls in self._word_classes:
            cls_counts = dict()
            for word_id, _ in cls:
                cls_counts[word_id] = counts[word_id]
            cls_total = sum(cls_counts.values())
            if cls_total > 0:
                for word_id, count in cls_counts.items():
                    cls.set_prob(word_id, float(count) / cls_total)

    def get_state(self, state):
        """Saves the vocabulary in a network state file.

        If there already is a vocabulary in the state, it will be replaced, so
        it has to have the same number of words.

        :type state: h5py.File
        :param state: HDF5 file for storing the neural network parameters
        """

        h5_vocabulary = state.require_group('vocabulary')

        if 'words' in h5_vocabulary:
            state['words'][:] = self.id_to_word
        else:
            str_dtype = h5py.special_dtype(vlen=str)
            h5_vocabulary.create_dataset('words',
                                         data=self.id_to_word,
                                         dtype=str_dtype)

        if 'classes' in h5_vocabulary:
            state['classes'][:] = self.word_id_to_class_id
        else:
            h5_vocabulary.create_dataset('classes', data=self.word_id_to_class_id)

        probs = [self._word_classes[class_id].get_prob(word_id)
                 for word_id, class_id in enumerate(self.word_id_to_class_id)]
        if 'probs' in h5_vocabulary:
            state['probs'][:] = probs
        else:
            h5_vocabulary.create_dataset('probs', data=probs)

    def num_words(self):
        """Returns the number of words in the vocabulary.

        :rtype: int
        :returns: the number of words in the vocabulary
        """

        return self.id_to_word.size

    def num_classes(self):
        """Returns the number of word classes.

        :rtype: int
        :returns: the number of words classes
        """

        return self._word_classes.size

    def words_to_ids(self, words):
        """Translates words into word IDs.

        :type words: list of strs
        :param words: a list of words

        :rtype: ndarray
        :returns: the given words translated into word IDs
        """

        unk_id = self.word_to_id['<unk>']
        result = numpy.zeros(len(words), dtype='int64')
        for index, word in enumerate(words):
            if word in self.word_to_id:
                result[index] = self.word_to_id[word]
            else:
                result[index] = unk_id
        return result

    def class_id_to_word_id(self, class_id):
        """Samples a word from the membership probability distribution of a
        class. (If classes are not used, returns the one word in the class.)

        :type class_id: int
        :param class_id: a class ID

        :rtype: int
        :returns: a word from the given class
        """

        return self._word_classes[class_id].sample()

    def word_ids_to_names(self, word_ids):
        """Translates word IDs into class / word names. If a class contains only
        one word, the name will be the word. Otherwise the name will be
        CLASS-12345, where 12345 is the internal class ID.

        :type word_ids: list of ints
        :param word_ids: a list of word IDs

        :rtype: list of strings
        :returns: a list of corresponding class / word names
        """

        class_ids = self.word_id_to_class_id[word_ids]
        return [self._class_name(self._word_classes[class_id])
                for class_id in class_ids]

    def _class_name(self, word_class):
        """If given class contains only one word, returns the word. Otherwicse
        returns CLASS-12345, where 12345 is the internal class ID.

        :type word_class: WordClass
        :param word_class: a word class object

        :rtype: str
        :returns: a name for the class
        """

        if len(word_class) == 1:
            word_id, _ = next(iter(word_class))
            return self.id_to_word[word_id]
        elif word_class.id is None:
            return 'CLASS'
        else:
            return 'CLASS-{:05d}'.format(word_class.id)

    def get_word_prob(self, word_id):
        """Returns the class membership probability of a word.

        :type word_id: int
        :param word_id: ID of a word

        :rtype: float
        :returns: the probability of the word within its class
        """

        class_id = self.word_id_to_class_id[word_id]
        word_class = self._word_classes[class_id]
        return word_class.get_prob(word_id)

    def get_class_memberships(self, word_ids):
        """Finds the classes and class membership probabilities given a matrix
        of word IDs.

        :type word_ids: ndarray
        :param word_ids: a matrix containing word IDs

        :rtype: tuple of ndarrays
        :returns: two matrices, the first one containing class IDs and the
                  second one containing class membership probabilities
        """

        class_ids = self.word_id_to_class_id[word_ids]
        word_classes = self._word_classes[class_ids]
        get_probs = numpy.vectorize(lambda wc, wid: wc.get_prob(wid))
        return class_ids, get_probs(word_classes, word_ids)

    def words(self):
        """A generator for iterating through the words in the vocabulary.

        :rtype: generator for str
        :returns: generates the next word in the vocabulary
        """

        for word in self.word_to_id.keys():
            yield word

    def __contains__(self, word):
        """Tests if ``word`` is included in the vocabulary.

        :type word: str
        :param word: a word

        :rtype: bool
        :returns: True if ``word`` is in the vocabulary, False otherwise.
        """

        return word in self.word_to_id
