#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
import mmap
import numpy
import theanolm
from theanolm.iterators.shufflingbatchiterator import find_sentence_starts

class TestVocabulary(unittest.TestCase):
    def setUp(self):
        script_path = os.path.dirname(os.path.realpath(__file__))
        sentences1_path = os.path.join(script_path, 'sentences1.txt')
        sentences2_path = os.path.join(script_path, 'sentences2.txt')
        vocabulary_path = os.path.join(script_path, 'vocabulary.txt')

        self.sentences1_file = open(sentences1_path)
        self.sentences2_file = open(sentences2_path)
        self.vocabulary_file = open(vocabulary_path)

    def tearDown(self):
        self.sentences1_file.close()
        self.sentences2_file.close()
        self.vocabulary_file.close()

    def test_from_file(self):
        self.vocabulary_file.seek(0)
        vocabulary = theanolm.Vocabulary.from_file(self.vocabulary_file, 'words')
        self.assertEqual(vocabulary.num_words(), 10 + 3)
        self.assertEqual(vocabulary.num_classes(), 10 + 3)

    def test_from_corpus(self):
        self.sentences1_file.seek(0)
        vocabulary = theanolm.Vocabulary.from_corpus([self.sentences1_file])
        self.assertEqual(vocabulary.num_words(), 10 + 3)
        self.assertEqual(vocabulary.num_classes(), 10 + 3)

        self.sentences1_file.seek(0)
        self.sentences2_file.seek(0)
        vocabulary = theanolm.Vocabulary.from_corpus([self.sentences1_file, self.sentences2_file], 3)
        self.assertEqual(vocabulary.num_words(), 10 + 3)
        self.assertEqual(vocabulary.num_classes(), 3 + 3)
        self.assertEqual(vocabulary.word_to_id['<s>'], 0)
        self.assertEqual(vocabulary.word_to_id['</s>'], 1)
        self.assertEqual(vocabulary.word_to_id['<unk>'], 2)
        self.assertEqual(vocabulary.word_to_class_id('<s>'), 0)
        self.assertEqual(vocabulary.word_to_class_id('</s>'), 1)
        self.assertEqual(vocabulary.word_to_class_id('<unk>'), 2)
        word_ids = set()
        class_ids = set()
        for word in vocabulary.words():
            if not word.startswith('<'):
                word_ids.add(vocabulary.word_to_id[word])
                class_ids.add(vocabulary.word_to_class_id(word))
        self.assertEqual(word_ids, set(range(3, 13)))
        self.assertEqual(class_ids, set(range(3, 6)))

if __name__ == '__main__':
    unittest.main()
