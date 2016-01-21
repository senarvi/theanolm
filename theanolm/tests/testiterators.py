#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
import mmap
import numpy
import theanolm
from theanolm.iterators.shufflingbatchiterator import find_sentence_starts

class TestIterators(unittest.TestCase):
    def setUp(self):
        script_path = os.path.dirname(os.path.realpath(__file__))
        sentences1_path = os.path.join(script_path, 'sentences1.txt')
        sentences2_path = os.path.join(script_path, 'sentences2.txt')
        dictionary_path = os.path.join(script_path, 'dictionary.txt')

        self.sentences1_file = open(sentences1_path)
        self.sentences2_file = open(sentences2_path)
        self.dictionary_file = open(dictionary_path)
        self.dictionary = theanolm.Dictionary(self.dictionary_file, 'words')

    def tearDown(self):
        self.sentences1_file.close()
        self.sentences2_file.close()
        self.dictionary_file.close()

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
        iterator = theanolm.ShufflingBatchIterator([self.sentences1_file, self.sentences2_file],
                                                   self.dictionary,
                                                   batch_size=2,
                                                   max_sequence_length=5)

        sentences1 = []
        for word_ids, probs, mask in iterator:
            for sequence in range(2):
                sequence_mask = numpy.array(mask)[:,sequence]
                sequence_word_ids = numpy.array(word_ids)[sequence_mask != 0,sequence]
                sentences1.append(' '.join(self.dictionary.ids_to_names(sequence_word_ids)))
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
        for word_ids, probs, mask in iterator:
            for sequence in range(2):
                sequence_mask = numpy.array(mask)[:,sequence]
                sequence_word_ids = numpy.array(word_ids)[sequence_mask != 0,sequence]
                sentences2.append(' '.join(self.dictionary.ids_to_names(sequence_word_ids)))
        sentences2_str = ' '.join(sentences2)
        sentences2_sorted_str = ' '.join(sorted(sentences2))
        self.assertEqual(sentences1_sorted_str, sentences2_sorted_str)
        self.assertNotEqual(sentences1_str, sentences2_str)

        # The current behaviour is to cut the sentences, so we always get 5
        # batches.
        iterator = theanolm.ShufflingBatchIterator([self.sentences1_file, self.sentences2_file],
                                                   self.dictionary,
                                                   batch_size=2,
                                                   max_sequence_length=4)
        self.assertEqual(len(iterator), 5)
        iterator = theanolm.ShufflingBatchIterator([self.sentences1_file, self.sentences2_file],
                                                   self.dictionary,
                                                   batch_size=2,
                                                   max_sequence_length=3)
        self.assertEqual(len(iterator), 5)

    def test_linear_batch_iterator(self):
        iterator = theanolm.LinearBatchIterator(self.sentences1_file,
                                                self.dictionary,
                                                batch_size=2,
                                                max_sequence_length=5)
        word_names = []
        for word_ids, probs, mask in iterator:
            mask = numpy.array(mask)
            word_ids = numpy.array(word_ids)
            for sequence in range(mask.shape[1]):
                sequence_mask = mask[:,sequence]
                sequence_word_ids = word_ids[sequence_mask != 0,sequence]
                word_names.extend(self.dictionary.ids_to_names(sequence_word_ids))
        corpus = ' '.join(word_names)
        self.assertEqual(corpus,
                         '<s> yksi kaksi </s> '
                         '<s> kolme neljä viisi </s> '
                         '<s> kuusi seitsemän kahdeksan </s> '
                         '<s> yhdeksän </s> '
                         '<s> kymmenen </s>')

if __name__ == '__main__':
    unittest.main()
