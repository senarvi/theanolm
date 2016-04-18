#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
import mmap
import numpy
import h5py
from theanolm import Vocabulary
from theanolm.iterators.shufflingbatchiterator import find_sentence_starts

class TestVocabulary(unittest.TestCase):
    def setUp(self):
        script_path = os.path.dirname(os.path.realpath(__file__))
        sentences1_path = os.path.join(script_path, 'sentences1.txt')
        sentences2_path = os.path.join(script_path, 'sentences2.txt')
        vocabulary_path = os.path.join(script_path, 'vocabulary.txt')
        classes_path = os.path.join(script_path, 'classes.txt')

        self.sentences1_file = open(sentences1_path)
        self.sentences2_file = open(sentences2_path)
        self.vocabulary_file = open(vocabulary_path)
        self.classes_file = open(classes_path)

    def tearDown(self):
        self.sentences1_file.close()
        self.sentences2_file.close()
        self.vocabulary_file.close()
        self.classes_file.close()

    def test_from_file(self):
        self.vocabulary_file.seek(0)
        vocabulary = Vocabulary.from_file(self.vocabulary_file, 'words')
        self.assertEqual(vocabulary.num_words(), 10 + 3)
        self.assertEqual(vocabulary.num_classes(), 10 + 3)

    def test_from_corpus(self):
        self.sentences1_file.seek(0)
        vocabulary = Vocabulary.from_corpus([self.sentences1_file])
        self.assertEqual(vocabulary.num_words(), 10 + 3)
        self.assertEqual(vocabulary.num_classes(), 10 + 3)

        self.sentences1_file.seek(0)
        self.sentences2_file.seek(0)
        vocabulary = Vocabulary.from_corpus([self.sentences1_file, self.sentences2_file], 3)
        self.assertEqual(vocabulary.num_words(), 10 + 3)
        self.assertEqual(vocabulary.num_classes(), 3 + 3)
        self.assertEqual(vocabulary.word_to_id['<s>'], 10)
        self.assertEqual(vocabulary.word_to_id['</s>'], 11)
        self.assertEqual(vocabulary.word_to_id['<unk>'], 12)
        self.assertEqual(vocabulary.word_to_class_id('<s>'), 3)
        self.assertEqual(vocabulary.word_to_class_id('</s>'), 4)
        self.assertEqual(vocabulary.word_to_class_id('<unk>'), 5)
        word_ids = set()
        class_ids = set()
        for word in vocabulary.words():
            if not word.startswith('<'):
                word_ids.add(vocabulary.word_to_id[word])
                class_ids.add(vocabulary.word_to_class_id(word))
        self.assertEqual(word_ids, set(range(10)))
        self.assertEqual(class_ids, set(range(3)))

    def test_from_state(self):
        self.classes_file.seek(0)
        vocabulary1 = Vocabulary.from_file(self.classes_file, 'srilm-classes')
        f = h5py.File('in-memory.h5', driver='core', backing_store=False)
        vocabulary1.get_state(f)
        vocabulary2 = Vocabulary.from_state(f)
        self.assertTrue(numpy.array_equal(vocabulary1.id_to_word, vocabulary2.id_to_word))
        self.assertDictEqual(vocabulary1.word_to_id, vocabulary2.word_to_id)
        self.assertTrue(numpy.array_equal(vocabulary1.word_id_to_class_id, vocabulary2.word_id_to_class_id))
        self.assertListEqual(vocabulary1._word_classes, vocabulary2._word_classes)

    def test_word_ids_to_names(self):
        self.classes_file.seek(0)
        vocabulary = Vocabulary.from_file(self.classes_file, 'srilm-classes')
        word_ids = [vocabulary.word_to_id['yksi'],
                    vocabulary.word_to_id['kaksi'],
                    vocabulary.word_to_id['kolme'],
                    vocabulary.word_to_id['neljä'],
                    vocabulary.word_to_id['viisi'],
                    vocabulary.word_to_id['kuusi'],
                    vocabulary.word_to_id['seitsemän'],
                    vocabulary.word_to_id['kahdeksan'],
                    vocabulary.word_to_id['yhdeksän'],
                    vocabulary.word_to_id['kymmenen']]
        names = vocabulary.word_ids_to_names(word_ids)
        self.assertEqual(names[0], 'yksi')
        self.assertEqual(names[1], 'kaksi')
        self.assertTrue(names[2].startswith('CLASS-'))
        self.assertEqual(names[2], names[3])
        self.assertEqual(names[4], 'viisi')
        self.assertTrue(names[5].startswith('CLASS-'))
        self.assertEqual(names[5], names[6])
        self.assertEqual(names[5], names[7])
        self.assertEqual(names[5], names[8])
        self.assertEqual(names[9], 'kymmenen')

if __name__ == '__main__':
    unittest.main()
