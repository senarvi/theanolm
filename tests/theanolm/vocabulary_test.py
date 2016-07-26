#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from os import path
import numpy
from numpy.testing import assert_almost_equal, assert_equal
import h5py
from theanolm import Vocabulary

class TestVocabulary(unittest.TestCase):
    def setUp(self):
        script_path = path.dirname(path.realpath(__file__))
        sentences1_path = path.join(script_path, 'sentences1.txt')
        sentences2_path = path.join(script_path, 'sentences2.txt')
        vocabulary_path = path.join(script_path, 'vocabulary.txt')
        classes_path = path.join(script_path, 'classes.txt')

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
        vocabulary = Vocabulary.from_corpus([self.sentences1_file,
                                             self.sentences2_file],
                                            3)
        self.assertEqual(vocabulary.num_words(), 10 + 3)
        self.assertEqual(vocabulary.num_classes(), 3 + 3)

        sos_id = vocabulary.word_to_id['<s>']
        eos_id = vocabulary.word_to_id['</s>']
        unk_id = vocabulary.word_to_id['<unk>']
        self.assertEqual(sos_id, 10)
        self.assertEqual(eos_id, 11)
        self.assertEqual(unk_id, 12)
        self.assertEqual(vocabulary.word_id_to_class_id[sos_id], 3)
        self.assertEqual(vocabulary.word_id_to_class_id[eos_id], 4)
        self.assertEqual(vocabulary.word_id_to_class_id[unk_id], 5)
        word_ids = set()
        class_ids = set()
        for word in vocabulary.words():
            if not word.startswith('<'):
                word_id = vocabulary.word_to_id[word]
                word_ids.add(word_id)
                class_ids.add(vocabulary.word_id_to_class_id[word_id])
        self.assertEqual(word_ids, set(range(10)))
        self.assertEqual(class_ids, set(range(3)))

    def test_from_state(self):
        self.classes_file.seek(0)
        vocabulary1 = Vocabulary.from_file(self.classes_file, 'srilm-classes')
        f = h5py.File('in-memory.h5', driver='core', backing_store=False)
        vocabulary1.get_state(f)
        vocabulary2 = Vocabulary.from_state(f)
        self.assertTrue(numpy.array_equal(vocabulary1.id_to_word,
                                          vocabulary2.id_to_word))
        self.assertDictEqual(vocabulary1.word_to_id, vocabulary2.word_to_id)
        self.assertTrue(numpy.array_equal(vocabulary1.word_id_to_class_id,
                                          vocabulary2.word_id_to_class_id))
        self.assertListEqual(list(vocabulary1._word_classes),
                             list(vocabulary2._word_classes))

    def test_compute_probs(self):
        self.classes_file.seek(0)
        vocabulary = Vocabulary.from_file(self.classes_file, 'srilm-classes')
        vocabulary.compute_probs([self.sentences1_file, self.sentences2_file])

        word_id = vocabulary.word_to_id['yksi']
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id), 1.0)
        word_id = vocabulary.word_to_id['kaksi']
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id), 1.0)
        word_id = vocabulary.word_to_id['kolme']
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id), 0.5)
        word_id = vocabulary.word_to_id['neljä']
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id), 0.5)
        word_id = vocabulary.word_to_id['viisi']
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id), 1.0)
        word_id = vocabulary.word_to_id['kuusi']
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id), 0.25)
        word_id = vocabulary.word_to_id['seitsemän']
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id), 0.25)
        word_id = vocabulary.word_to_id['kahdeksan']
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id), 0.25)
        word_id = vocabulary.word_to_id['yhdeksän']
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id), 0.25)
        word_id = vocabulary.word_to_id['kymmenen']
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id), 1.0)

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

    def test_get_class_memberships(self):
        vocabulary = Vocabulary.from_file(self.classes_file, 'srilm-classes')
        word_ids = numpy.array([vocabulary.word_to_id['yksi'],
                                vocabulary.word_to_id['kaksi'],
                                vocabulary.word_to_id['kolme'],
                                vocabulary.word_to_id['neljä'],
                                vocabulary.word_to_id['viisi'],
                                vocabulary.word_to_id['kuusi'],
                                vocabulary.word_to_id['seitsemän'],
                                vocabulary.word_to_id['kahdeksan'],
                                vocabulary.word_to_id['yhdeksän'],
                                vocabulary.word_to_id['kymmenen']])
        class_ids, probs = vocabulary.get_class_memberships(word_ids)
        assert_equal(class_ids, vocabulary.word_id_to_class_id[word_ids])
        assert_equal(class_ids[3], vocabulary.word_id_to_class_id[word_ids[3]])
        assert_almost_equal(probs, [1.0,
                                    1.0,
                                    0.599 / (0.599 + 0.400),
                                    0.400 / (0.599 + 0.400),
                                    1.0,
                                    0.281 / (0.281 + 0.226 + 0.262 + 0.228),
                                    0.226 / (0.281 + 0.226 + 0.262 + 0.228),
                                    0.262 / (0.281 + 0.226 + 0.262 + 0.228),
                                    0.228 / (0.281 + 0.226 + 0.262 + 0.228),
                                    1.0])

if __name__ == '__main__':
    unittest.main()
