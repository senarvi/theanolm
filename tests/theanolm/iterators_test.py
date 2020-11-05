#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
import mmap

import numpy
from numpy.testing import assert_equal

from theanolm import Vocabulary
from theanolm.parsing import LinearBatchIterator, ScoringBatchIterator
from theanolm.parsing import ShufflingBatchIterator
from theanolm.parsing.functions import find_sentence_starts

class TestIterators(unittest.TestCase):
    def setUp(self):
        """
        Set sentences tomodels.

        Args:
            self: (todo): write your description
        """
        script_path = os.path.dirname(os.path.realpath(__file__))
        sentences1_path = os.path.join(script_path, 'sentences1.txt')
        sentences2_path = os.path.join(script_path, 'sentences2.txt')
        sentences3_path = os.path.join(script_path, 'sentences3.txt')
        vocabulary_path = os.path.join(script_path, 'vocabulary.txt')

        self.sentences1_file = open(sentences1_path)
        self.sentences2_file = open(sentences2_path)
        self.sentences3_file = open(sentences3_path)
        self.vocabulary_file = open(vocabulary_path)
        self.vocabulary = Vocabulary.from_file(self.vocabulary_file, 'words')
        self.vocabulary_file.seek(0)
        self.shortlist_vocabulary = \
            Vocabulary.from_file(self.vocabulary_file, 'words',
                                 oos_words=['yksitoista'])

    def tearDown(self):
        """
        Tear down sentences.

        Args:
            self: (todo): write your description
        """
        self.sentences1_file.close()
        self.sentences2_file.close()
        self.sentences3_file.close()
        self.vocabulary_file.close()

    def test_find_sentence_starts(self):
        """
        Compare two sentence sentences that sentence.

        Args:
            self: (todo): write your description
        """
        sentences1_mmap = mmap.mmap(self.sentences1_file.fileno(),
                                    0,
                                    access=mmap.ACCESS_READ)
        sentence_starts = find_sentence_starts(sentences1_mmap)
        self.sentences1_file.seek(sentence_starts[0])
        self.assertEqual(self.sentences1_file.readline(), 'yksi kaksi\n')
        self.sentences1_file.seek(sentence_starts[1])
        self.assertEqual(self.sentences1_file.readline(), 'kolme neljä viisi\n')
        self.sentences1_file.seek(sentence_starts[2])
        self.assertEqual(self.sentences1_file.readline(), 'kuusi seitsemän kahdeksan\n')
        self.sentences1_file.seek(sentence_starts[3])
        self.assertEqual(self.sentences1_file.readline(), 'yhdeksän\n')
        self.sentences1_file.seek(sentence_starts[4])
        self.assertEqual(self.sentences1_file.readline(), 'kymmenen\n')
        self.sentences1_file.seek(0)

        sentences2_mmap = mmap.mmap(self.sentences2_file.fileno(),
                                    0,
                                    access=mmap.ACCESS_READ)
        sentence_starts = find_sentence_starts(sentences2_mmap)
        self.sentences2_file.seek(sentence_starts[0])
        self.assertEqual(self.sentences2_file.readline(), 'kymmenen yhdeksän\n')
        self.sentences2_file.seek(sentence_starts[1])
        self.assertEqual(self.sentences2_file.readline(), 'kahdeksan seitsemän kuusi\n')
        self.sentences2_file.seek(sentence_starts[2])
        self.assertEqual(self.sentences2_file.readline(), 'viisi\n')
        self.sentences2_file.seek(sentence_starts[3])
        self.assertEqual(self.sentences2_file.readline(), 'neljä\n')
        self.sentences2_file.seek(sentence_starts[4])
        self.assertEqual(self.sentences2_file.readline(), 'kolme kaksi yksi\n')
        self.sentences2_file.seek(0)

    def test_shuffling_batch_iterator(self):
        """
        Create a batch of sentences.

        Args:
            self: (todo): write your description
        """
        iterator = ShufflingBatchIterator([self.sentences1_file,
                                           self.sentences2_file],
                                          [],
                                          self.vocabulary,
                                          batch_size=2,
                                          max_sequence_length=5)

        sentences1 = []
        files1 = []
        for word_ids, file_ids, mask in iterator:
            class_ids = self.vocabulary.word_id_to_class_id[word_ids]
            for sequence in range(2):
                sequence_mask = mask[:,sequence]
                sequence_word_ids = word_ids[sequence_mask != 0,sequence]
                sequence_class_ids = class_ids[sequence_mask != 0,sequence]
                sequence_file_ids = file_ids[sequence_mask != 0,sequence]
                assert_equal(sequence_word_ids, sequence_class_ids)
                sentences1.append(' '.join(self.vocabulary.id_to_word[sequence_word_ids]))
                files1.extend(sequence_file_ids)
        self.assertEqual(files1.count(0), 20)
        self.assertEqual(files1.count(1), 20)
        sentences1_str = ' '.join(sentences1)
        sentences1_sorted_str = ' '.join(sorted(sentences1))
        self.assertEqual(sentences1_sorted_str,
                         '<s> kahdeksan seitsemän kuusi </s> '
                         '<s> kolme kaksi yksi </s> '
                         '<s> kolme neljä viisi </s> '
                         '<s> kuusi seitsemän kahdeksan </s> '
                         '<s> kymmenen </s> '
                         '<s> kymmenen yhdeksän </s> '
                         '<s> neljä </s> '
                         '<s> viisi </s> '
                         '<s> yhdeksän </s> '
                         '<s> yksi kaksi </s>')
        self.assertEqual(len(iterator), 5)

        sentences2 = []
        files2 = []
        for word_ids, file_ids, mask in iterator:
            class_ids = self.vocabulary.word_id_to_class_id[word_ids]
            for sequence in range(2):
                sequence_mask = mask[:,sequence]
                sequence_word_ids = word_ids[sequence_mask != 0,sequence]
                sequence_class_ids = class_ids[sequence_mask != 0,sequence]
                sequence_file_ids = file_ids[sequence_mask != 0,sequence]
                assert_equal(sequence_word_ids, sequence_class_ids)
                sentences2.append(' '.join(self.vocabulary.id_to_word[sequence_word_ids]))
                files2.extend(sequence_file_ids)
        self.assertCountEqual(sentences1, sentences2)
        self.assertCountEqual(files1, files2)
        self.assertTrue(sentences1 != sentences2)
        self.assertTrue(files1 != files2)

        # The sentences are wraped so that we get more sequences if we limit
        # the maximum length. Sequences shorter than two words will be ignored.
        iterator = ShufflingBatchIterator([self.sentences1_file,
                                           self.sentences2_file],
                                          [],
                                          self.vocabulary,
                                          batch_size=2,
                                          max_sequence_length=4)
        self.assertEqual(len(iterator), 5)
        iterator = ShufflingBatchIterator([self.sentences1_file,
                                           self.sentences2_file],
                                          [],
                                          self.vocabulary,
                                          batch_size=2,
                                          max_sequence_length=3)
        self.assertEqual(len(iterator), 7)

        # Sample 2 and 4 sentences (40 % and 80 %).
        iterator = ShufflingBatchIterator([self.sentences1_file,
                                           self.sentences2_file],
                                          [0.4, 0.8],
                                          self.vocabulary,
                                          batch_size=1,
                                          max_sequence_length=5)
        self.assertEqual(len(iterator), 2 + 4)

        # Make sure there are no duplicates.
        self.assertSetEqual(set(iterator._order),
                            set(numpy.unique(iterator._order)))
        self.assertEqual(numpy.count_nonzero(iterator._order <= 4), 2)
        self.assertEqual(numpy.count_nonzero(iterator._order >= 5), 4)

        # Use shortlist and don't map OOS words to <unk>.
        iterator = ShufflingBatchIterator([self.sentences1_file,
                                           self.sentences2_file,
                                           self.sentences3_file],
                                          [],
                                          self.shortlist_vocabulary,
                                          batch_size=2,
                                          map_oos_to_unk=False)
        word_counts = self._compute_word_counts(iterator)
        self._assert_oos_counts(word_counts)

        # Use shortlist and map OOS words to <unk>.
        iterator = ShufflingBatchIterator([self.sentences1_file,
                                           self.sentences2_file,
                                           self.sentences3_file],
                                          [],
                                          self.shortlist_vocabulary,
                                          batch_size=2,
                                          map_oos_to_unk=True)
        word_counts = self._compute_word_counts(iterator)
        self._assert_shortlist_counts(word_counts)

    def test_linear_batch_iterator(self):
        """
        Loads a batch of sentences.

        Args:
            self: (todo): write your description
        """
        iterator = LinearBatchIterator(self.sentences1_file,
                                                self.vocabulary,
                                                batch_size=2,
                                                max_sequence_length=5)
        words = []
        for word_ids, file_ids, mask in iterator:
            class_ids = self.vocabulary.word_id_to_class_id[word_ids]
            assert_equal(word_ids, class_ids)
            assert_equal(file_ids, 0)
            for sequence in range(mask.shape[1]):
                sequence_mask = mask[:,sequence]
                sequence_word_ids = word_ids[sequence_mask != 0,sequence]
                words.extend(self.vocabulary.id_to_word[sequence_word_ids])
        self.assertEqual(' '.join(words),
                         '<s> yksi kaksi </s> '
                         '<s> kolme neljä viisi </s> '
                         '<s> kuusi seitsemän kahdeksan </s> '
                         '<s> yhdeksän </s> '
                         '<s> kymmenen </s>')

        iterator = LinearBatchIterator([self.sentences1_file,
                                        self.sentences2_file],
                                       self.vocabulary,
                                       batch_size=2,
                                       max_sequence_length=5)
        words = []
        all_file_ids = []
        for word_ids, file_ids, mask in iterator:
            class_ids = self.vocabulary.word_id_to_class_id[word_ids]
            assert_equal(word_ids, class_ids)
            for sequence in range(mask.shape[1]):
                sequence_mask = mask[:,sequence]
                sequence_word_ids = word_ids[sequence_mask != 0,sequence]
                words.extend(self.vocabulary.id_to_word[sequence_word_ids])
                sequence_file_ids = file_ids[sequence_mask != 0,sequence]
                all_file_ids.extend(sequence_file_ids)
        self.assertEqual(' '.join(words),
                         '<s> yksi kaksi </s> '
                         '<s> kolme neljä viisi </s> '
                         '<s> kuusi seitsemän kahdeksan </s> '
                         '<s> yhdeksän </s> '
                         '<s> kymmenen </s> '
                         '<s> kymmenen yhdeksän </s> '
                         '<s> kahdeksan seitsemän kuusi </s> '
                         '<s> viisi </s> '
                         '<s> neljä </s> '
                         '<s> kolme kaksi yksi </s>')
        assert_equal(all_file_ids, [0, 0, 0, 0,
                                    0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0,
                                    0, 0, 0,
                                    0, 0, 0,
                                    1, 1, 1, 1,
                                    1, 1, 1, 1, 1,
                                    1, 1, 1,
                                    1, 1, 1,
                                    1, 1, 1, 1, 1])

        # Use shortlist and don't map OOS words to <unk>.
        iterator = LinearBatchIterator([self.sentences1_file,
                                        self.sentences2_file,
                                        self.sentences3_file],
                                       self.shortlist_vocabulary,
                                       batch_size=2,
                                       map_oos_to_unk=False)
        word_counts = self._compute_word_counts(iterator)
        self._assert_oos_counts(word_counts)

        # Use shortlist and map OOS words to <unk>.
        iterator = LinearBatchIterator([self.sentences1_file,
                                        self.sentences2_file,
                                        self.sentences3_file],
                                       self.shortlist_vocabulary,
                                       batch_size=2,
                                       map_oos_to_unk=True)
        word_counts = self._compute_word_counts(iterator)
        self._assert_shortlist_counts(word_counts)

    def test_scoring_batch_iterator(self):
        """
        Parameters ---------- batch_size : ints

        Args:
            self: (todo): write your description
        """
        iterator = ScoringBatchIterator(self.sentences1_file,
                                        self.vocabulary,
                                        batch_size=2,
                                        max_sequence_length=5)
        all_words = []
        for word_ids, words, mask in iterator:
            class_ids = self.vocabulary.word_id_to_class_id[word_ids]
            assert_equal(word_ids, class_ids)
            for sequence in range(mask.shape[1]):
                sequence_words = words[sequence]
                all_words.extend(sequence_words)
        self.assertEqual(' '.join(all_words),
                         '<s> yksi kaksi </s> '
                         '<s> kolme neljä viisi </s> '
                         '<s> kuusi seitsemän kahdeksan </s> '
                         '<s> yhdeksän </s> '
                         '<s> kymmenen </s>')

        iterator = ScoringBatchIterator([self.sentences1_file,
                                         self.sentences2_file],
                                        self.vocabulary,
                                        batch_size=2,
                                        max_sequence_length=5)
        all_words = []
        for word_ids, words, mask in iterator:
            class_ids = self.vocabulary.word_id_to_class_id[word_ids]
            assert_equal(word_ids, class_ids)
            for sequence in range(mask.shape[1]):
                sequence_words = words[sequence]
                all_words.extend(sequence_words)
        self.assertEqual(' '.join(all_words),
                         '<s> yksi kaksi </s> '
                         '<s> kolme neljä viisi </s> '
                         '<s> kuusi seitsemän kahdeksan </s> '
                         '<s> yhdeksän </s> '
                         '<s> kymmenen </s> '
                         '<s> kymmenen yhdeksän </s> '
                         '<s> kahdeksan seitsemän kuusi </s> '
                         '<s> viisi </s> '
                         '<s> neljä </s> '
                         '<s> kolme kaksi yksi </s>')

        # Use shortlist and don't map OOS words to <unk>.
        iterator = ScoringBatchIterator([self.sentences1_file,
                                         self.sentences2_file,
                                         self.sentences3_file],
                                        self.shortlist_vocabulary,
                                        batch_size=2,
                                        map_oos_to_unk=False)
        word_counts = self._compute_word_counts(iterator)
        self._assert_oos_counts(word_counts)

        # Use shortlist and map OOS words to <unk>.
        iterator = ScoringBatchIterator([self.sentences1_file,
                                         self.sentences2_file,
                                         self.sentences3_file],
                                        self.shortlist_vocabulary,
                                        batch_size=2,
                                        map_oos_to_unk=True)
        word_counts = self._compute_word_counts(iterator)
        self._assert_shortlist_counts(word_counts)

    def _compute_word_counts(self, iterator):
        """Compute words counts using ``iterator``.
        """

        result = numpy.zeros(self.shortlist_vocabulary.num_words(),
                             dtype='int64')
        for word_ids, file_ids, mask in iterator:
            word_ids = word_ids[mask == 1]
            numpy.add.at(result, word_ids, 1)
        return result

    def _assert_oos_counts(self, word_counts):
        """When not mapping OOS words to ``<unk>``, the shortlist words appear
        three times each, and the OOS word and ``<unk>`` appear once. Assert
        that ``word_counts`` matches this.
        """

        for word in ['yksi', 'kaksi', 'kolme', 'neljä', 'viisi', 'kuusi',
                     'seitsemän', 'kahdeksan', 'yhdeksän', 'kymmenen']:
            word_id = self.shortlist_vocabulary.word_to_id[word]
            self.assertEqual(word_counts[word_id], 3)
        word_id = self.shortlist_vocabulary.word_to_id['yksitoista']
        self.assertEqual(word_counts[word_id], 1)
        word_id = self.shortlist_vocabulary.word_to_id['<unk>']
        self.assertEqual(word_counts[word_id], 1)
        word_id = self.shortlist_vocabulary.word_to_id['<s>']
        self.assertEqual(word_counts[word_id], 15)
        word_id = self.shortlist_vocabulary.word_to_id['</s>']
        self.assertEqual(word_counts[word_id], 15)

    def _assert_shortlist_counts(self, word_counts):
        """When mapping OOS words to ``<unk>``, the shortlist words appear three
        times each, the OOS word appears zero times, and ``<unk>`` appear twice.
        Assert that ``word_counts`` matches this.
        """

        for word in ['yksi', 'kaksi', 'kolme', 'neljä', 'viisi', 'kuusi',
                     'seitsemän', 'kahdeksan', 'yhdeksän', 'kymmenen']:
            word_id = self.shortlist_vocabulary.word_to_id[word]
            self.assertEqual(word_counts[word_id], 3)
        word_id = self.shortlist_vocabulary.word_to_id['yksitoista']
        self.assertEqual(word_counts[word_id], 0)
        word_id = self.shortlist_vocabulary.word_to_id['<unk>']
        self.assertEqual(word_counts[word_id], 2)
        word_id = self.shortlist_vocabulary.word_to_id['<s>']
        self.assertEqual(word_counts[word_id], 15)
        word_id = self.shortlist_vocabulary.word_to_id['</s>']
        self.assertEqual(word_counts[word_id], 15)

if __name__ == '__main__':
    unittest.main()
