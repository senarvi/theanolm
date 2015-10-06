#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
import theanolm

class TestIterators(unittest.TestCase):
    def setUp(self):
        script_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(script_path, 'sentences.txt')
        self.file = open(data_path)

    def test_find_sentence_starts(self):
        sentence_starts = theanolm.find_sentence_starts(self.file)
        self.file.seek(sentence_starts[0])
        self.assertEqual(self.file.readline(), 'yksi kaksi\n')
        self.file.seek(sentence_starts[1])
        self.assertEqual(self.file.readline(), 'kolme neljä viisi\n')
        self.file.seek(sentence_starts[2])
        self.assertEqual(self.file.readline(), 'kuusi seitsemän kahdeksan\n')
        self.file.seek(sentence_starts[3])
        self.assertEqual(self.file.readline(), 'yhdeksän\n')

if __name__ == '__main__':
    unittest.main()
