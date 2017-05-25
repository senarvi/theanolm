#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from os import path

import h5py
import numpy
from numpy.testing import assert_almost_equal, assert_equal

from theanolm.vocabulary import Vocabulary, compute_word_counts

class TestVocabulary(unittest.TestCase):
    def setUp(self):
        script_path = path.dirname(path.realpath(__file__))
        sentences1_path = path.join(script_path, 'sentences1.txt')
        sentences2_path = path.join(script_path, 'sentences2.txt')
        sentences3_path = path.join(script_path, 'sentences3.txt')
        vocabulary_path = path.join(script_path, 'vocabulary.txt')
        classes_path = path.join(script_path, 'classes.txt')

        self.sentences1_file = open(sentences1_path)
        self.sentences2_file = open(sentences2_path)
        self.sentences3_file = open(sentences3_path)
        self.vocabulary_file = open(vocabulary_path)
        self.classes_file = open(classes_path)

    def tearDown(self):
        self.sentences1_file.close()
        self.sentences2_file.close()
        self.sentences3_file.close()
        self.vocabulary_file.close()
        self.classes_file.close()

    def test_from_file(self):
        self.vocabulary_file.seek(0)
        vocabulary = Vocabulary.from_file(self.vocabulary_file, 'words')
        self.assertEqual(vocabulary.num_words(), 10 + 3)
        self.assertEqual(vocabulary.num_shortlist_words(), 10 + 3)
        self.assertEqual(vocabulary.num_classes(), 10 + 3)

        oos_words = ['yksi', 'kaksi', 'yksitoista', 'kaksitoista']
        self.vocabulary_file.seek(0)
        vocabulary = Vocabulary.from_file(self.vocabulary_file, 'words',
                                          oos_words=oos_words)
        self.assertEqual(vocabulary.num_words(), 12 + 3)
        self.assertEqual(vocabulary.num_shortlist_words(), 10 + 3)
        self.assertEqual(vocabulary.num_classes(), 10 + 3)

    def test_from_word_counts(self):
        self.sentences1_file.seek(0)
        word_counts = compute_word_counts([self.sentences1_file])
        vocabulary = Vocabulary.from_word_counts(word_counts)
        self.assertEqual(vocabulary.num_words(), 10 + 3)
        self.assertEqual(vocabulary.num_shortlist_words(), 10 + 3)
        self.assertEqual(vocabulary.num_normal_classes, 10)
        self.assertEqual(vocabulary.num_classes(), 10 + 3)

        self.sentences1_file.seek(0)
        self.sentences2_file.seek(0)
        word_counts = compute_word_counts([self.sentences1_file,
                                           self.sentences2_file])
        vocabulary = Vocabulary.from_word_counts(word_counts, 3)
        self.assertEqual(vocabulary.num_words(), 10 + 3)
        self.assertEqual(vocabulary.num_shortlist_words(), 10 + 3)
        self.assertEqual(vocabulary.num_normal_classes, 3)
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
        word_counts = compute_word_counts([self.sentences1_file,
                                           self.sentences2_file])
        vocabulary1.compute_probs(word_counts)

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
        self.assertTrue(numpy.array_equal(vocabulary1._unigram_probs,
                                          vocabulary2._unigram_probs))

    def test_class_ids(self):
        self.classes_file.seek(0)
        vocabulary = Vocabulary.from_file(self.classes_file, 'srilm-classes')
        word_id = vocabulary.word_to_id['yksi']
        yksi_class_id = vocabulary.word_id_to_class_id[word_id]
        word_id = vocabulary.word_to_id['kaksi']
        kaksi_class_id = vocabulary.word_id_to_class_id[word_id]
        word_id = vocabulary.word_to_id['kolme']
        kolme_class_id = vocabulary.word_id_to_class_id[word_id]
        word_id = vocabulary.word_to_id['neljä']
        nelja_class_id = vocabulary.word_id_to_class_id[word_id]
        word_id = vocabulary.word_to_id['</s>']
        eos_class_id = vocabulary.word_id_to_class_id[word_id]
        self.assertNotEqual(yksi_class_id, kaksi_class_id)
        self.assertEqual(kolme_class_id, nelja_class_id)
        self.assertNotEqual(kolme_class_id, eos_class_id)
        self.assertEqual(kaksi_class_id, eos_class_id)

    def test_compute_probs(self):
        self.classes_file.seek(0)
        vocabulary = Vocabulary.from_file(self.classes_file, 'srilm-classes')
        word_counts = compute_word_counts([self.sentences1_file,
                                           self.sentences2_file])
        vocabulary.compute_probs(word_counts)

        # 10 * </s> + 20 words. <s> is excluded.
        total_count = 30.0
        word_id = vocabulary.word_to_id['yksi']
        self.assertAlmostEqual(vocabulary._unigram_probs[word_id],
                               2.0 / total_count)
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id), 1.0)
        word_id = vocabulary.word_to_id['kaksi']
        self.assertAlmostEqual(vocabulary._unigram_probs[word_id],
                               2.0 / total_count)
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id), 2.0 / 12.0)
        word_id = vocabulary.word_to_id['kolme']
        self.assertAlmostEqual(vocabulary._unigram_probs[word_id],
                               2.0 / total_count)
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id), 0.5)
        word_id = vocabulary.word_to_id['neljä']
        self.assertAlmostEqual(vocabulary._unigram_probs[word_id],
                               2.0 / total_count)
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id), 0.5)
        word_id = vocabulary.word_to_id['viisi']
        self.assertAlmostEqual(vocabulary._unigram_probs[word_id],
                               2.0 / total_count)
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id), 1.0)
        word_id = vocabulary.word_to_id['kuusi']
        self.assertAlmostEqual(vocabulary._unigram_probs[word_id],
                               2.0 / total_count)
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id), 0.25)
        word_id = vocabulary.word_to_id['seitsemän']
        self.assertAlmostEqual(vocabulary._unigram_probs[word_id],
                               2.0 / total_count)
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id), 0.25)
        word_id = vocabulary.word_to_id['kahdeksan']
        self.assertAlmostEqual(vocabulary._unigram_probs[word_id],
                               2.0 / total_count)
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id), 0.25)
        word_id = vocabulary.word_to_id['yhdeksän']
        self.assertAlmostEqual(vocabulary._unigram_probs[word_id],
                               2.0 / total_count)
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id), 0.25)
        word_id = vocabulary.word_to_id['kymmenen']
        self.assertAlmostEqual(vocabulary._unigram_probs[word_id],
                               2.0 / total_count)
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id), 1.0)
        word_id = vocabulary.word_to_id['<s>']
        self.assertAlmostEqual(vocabulary._unigram_probs[word_id], 0.0)
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id), 1.0)
        word_id = vocabulary.word_to_id['</s>']
        self.assertAlmostEqual(vocabulary._unigram_probs[word_id],
                               10.0 / total_count)
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id),
                               10.0 / 12.0)
        word_id = vocabulary.word_to_id['<unk>']
        self.assertAlmostEqual(vocabulary._unigram_probs[word_id], 0.0)
        self.assertAlmostEqual(vocabulary.get_word_prob(word_id), 1.0)

    def test_get_class_memberships(self):
        vocabulary = Vocabulary.from_file(self.classes_file, 'srilm-classes',
                                          oos_words=['yksitoista'])
        word_ids = numpy.array([vocabulary.word_to_id['yksi'],
                                vocabulary.word_to_id['kaksi'],
                                vocabulary.word_to_id['kolme'],
                                vocabulary.word_to_id['neljä'],
                                vocabulary.word_to_id['viisi'],
                                vocabulary.word_to_id['kuusi'],
                                vocabulary.word_to_id['seitsemän'],
                                vocabulary.word_to_id['kahdeksan'],
                                vocabulary.word_to_id['yhdeksän'],
                                vocabulary.word_to_id['kymmenen'],
                                vocabulary.word_to_id['<s>'],
                                vocabulary.word_to_id['</s>'],
                                vocabulary.word_to_id['<unk>']])
        class_ids, probs = vocabulary.get_class_memberships(word_ids)
        assert_equal(class_ids, vocabulary.word_id_to_class_id[word_ids])
        assert_almost_equal(probs, [1.0,
                                    0.999,
                                    0.599 / (0.599 + 0.400),
                                    0.400 / (0.599 + 0.400),
                                    1.0,
                                    0.281 / (0.281 + 0.226 + 0.262 + 0.228),
                                    0.226 / (0.281 + 0.226 + 0.262 + 0.228),
                                    0.262 / (0.281 + 0.226 + 0.262 + 0.228),
                                    0.228 / (0.281 + 0.226 + 0.262 + 0.228),
                                    1.0,
                                    1.0,
                                    0.001,
                                    1.0])

        word_counts = compute_word_counts([self.sentences3_file])
        vocabulary.compute_probs(word_counts)
        class_ids, probs = vocabulary.get_class_memberships(word_ids)
        assert_almost_equal(probs, [1.0,
                                    1.0 / 6.0,
                                    0.5,
                                    0.5,
                                    1.0,
                                    0.25,
                                    0.25,
                                    0.25,
                                    0.25,
                                    1.0,
                                    1.0,
                                    5.0 / 6.0,
                                    1.0])

    def test_get_oos_logprobs(self):
        oos_words = ['yksitoista', 'kaksitoista']
        self.vocabulary_file.seek(0)
        vocabulary = Vocabulary.from_file(self.vocabulary_file, 'words',
                                          oos_words=oos_words)
        word_counts = {'yksi': 1, 'kaksi': 2, 'kolme': 3, 'neljä': 4,
                       'viisi': 5, 'kuusi': 6, 'seitsemän': 7, 'kahdeksan': 8,
                       'yhdeksän': 9, 'kymmenen': 10, '<s>': 11, '</s>': 12,
                       '<unk>': 13, 'yksitoista': 3, 'kaksitoista': 7}
        vocabulary.compute_probs(word_counts)
        oos_logprobs = vocabulary.get_oos_logprobs()
        assert_almost_equal(oos_logprobs, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                           numpy.log(0.3), numpy.log(0.7)])

if __name__ == '__main__':
    unittest.main()
