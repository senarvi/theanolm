#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
import mmap
import numpy
from numpy.testing import assert_equal
import theanolm
from theanolm.iterators.shufflingbatchiterator import find_sentence_starts

class TestIterators(unittest.TestCase):
    def setUp(self):
        script_path = os.path.dirname(os.path.realpath(__file__))
        sentences1_path = os.path.join(script_path, 'sentences1.txt')
        sentences2_path = os.path.join(script_path, 'sentences2.txt')
        vocabulary_path = os.path.join(script_path, 'vocabulary.txt')

        self.sentences1_file = open(sentences1_path)
        self.sentences2_file = open(sentences2_path)
        self.vocabulary_file = open(vocabulary_path)
        self.vocabulary = theanolm.Vocabulary.from_file(self.vocabulary_file, 'words')

    def tearDown(self):
        self.sentences1_file.close()
        self.sentences2_file.close()
        self.vocabulary_file.close()

    def test_find_sentence_starts(self):
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
        iterator = theanolm.ShufflingBatchIterator([self.sentences1_file,
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
                sentences1.append(' '.join(self.vocabulary.word_ids_to_names(sequence_word_ids)))
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
                sentences2.append(' '.join(self.vocabulary.word_ids_to_names(sequence_word_ids)))
                files2.extend(sequence_file_ids)
        self.assertCountEqual(sentences1, sentences2)
        self.assertCountEqual(files1, files2)
        self.assertTrue(sentences1 != sentences2)
        self.assertTrue(files1 != files2)

        # The current behaviour is to cut the sentences, so we always get 5
        # batches regardless of the maximum sequence length.
        iterator = theanolm.ShufflingBatchIterator([self.sentences1_file,
                                                    self.sentences2_file],
                                                   [],
                                                   self.vocabulary,
                                                   batch_size=2,
                                                   max_sequence_length=4)
        self.assertEqual(len(iterator), 5)
        iterator = theanolm.ShufflingBatchIterator([self.sentences1_file,
                                                    self.sentences2_file],
                                                   [],
                                                   self.vocabulary,
                                                   batch_size=2,
                                                   max_sequence_length=3)
        self.assertEqual(len(iterator), 5)

        # Sample 2 and 4 sentences (40 % and 80 %).
        iterator = theanolm.ShufflingBatchIterator([self.sentences1_file,
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

    def test_linear_batch_iterator(self):
        iterator = theanolm.LinearBatchIterator(self.sentences1_file,
                                                self.vocabulary,
                                                batch_size=2,
                                                max_sequence_length=5)
        word_names = []
        for word_ids, file_ids, mask in iterator:
            class_ids = self.vocabulary.word_id_to_class_id[word_ids]
            assert_equal(word_ids, class_ids)
            assert_equal(file_ids, 0)
            for sequence in range(mask.shape[1]):
                sequence_mask = mask[:,sequence]
                sequence_word_ids = word_ids[sequence_mask != 0,sequence]
                word_names.extend(self.vocabulary.word_ids_to_names(sequence_word_ids))
        corpus = ' '.join(word_names)
        self.assertEqual(corpus,
                         '<s> yksi kaksi </s> '
                         '<s> kolme neljä viisi </s> '
                         '<s> kuusi seitsemän kahdeksan </s> '
                         '<s> yhdeksän </s> '
                         '<s> kymmenen </s>')

if __name__ == '__main__':
    unittest.main()
