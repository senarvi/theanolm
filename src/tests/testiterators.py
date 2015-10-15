#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
import mmap
import theanolm

class TestIterators(unittest.TestCase):
    def setUp(self):
        script_path = os.path.dirname(os.path.realpath(__file__))
        sentences_path = os.path.join(script_path, 'sentences.txt')
        dictionary_path = os.path.join(script_path, 'dictionary.txt')

        self.sentences_file = open(sentences_path)
        self.dictionary_file = open(dictionary_path)
        self.dictionary = theanolm.Dictionary(self.dictionary_file, 'words')

    def tearDown(self):
        self.sentences_file.close()
        self.dictionary_file.close()

    def test_find_sentence_starts(self):
        sentences_mmap = mmap.mmap(self.sentences_file.fileno(),
                                   0,
                                   access=mmap.ACCESS_READ)
        sentence_starts = theanolm.find_sentence_starts(sentences_mmap)
        self.sentences_file.seek(sentence_starts[0])
        self.assertEqual(self.sentences_file.readline(), 'yksi kaksi\n')
        self.sentences_file.seek(sentence_starts[1])
        self.assertEqual(self.sentences_file.readline(), 'kolme neljä viisi\n')
        self.sentences_file.seek(sentence_starts[2])
        self.assertEqual(self.sentences_file.readline(), 'kuusi seitsemän kahdeksan\n')
        self.sentences_file.seek(sentence_starts[3])
        self.assertEqual(self.sentences_file.readline(), 'yhdeksän\n')
        self.sentences_file.seek(sentence_starts[4])
        self.assertEqual(self.sentences_file.readline(), 'kymmenen\n')

        self.sentences_file.seek(0)
        iter = theanolm.ShufflingBatchIterator(self.sentences_file,
                                               self.dictionary,
                                               sentence_starts,
                                               batch_size=2,
                                               max_sequence_length=3)
        self.assertEqual(len(iter), 3)
        iter = theanolm.ShufflingBatchIterator(self.sentences_file,
                                               self.dictionary,
                                               sentence_starts,
                                               batch_size=2,
                                               max_sequence_length=2)
        self.assertEqual(len(iter), 2)
        iter = theanolm.ShufflingBatchIterator(self.sentences_file,
                                               self.dictionary,
                                               sentence_starts,
                                               batch_size=2,
                                               max_sequence_length=1)
        self.assertEqual(len(iter), 1)

if __name__ == '__main__':
    unittest.main()
