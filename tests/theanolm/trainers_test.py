#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
from theanolm.trainers import BasicTrainer

class DummyTrainer(object):
    pass

class TestIterators(unittest.TestCase):
    def setUp(self):
        self.dummy_trainer = DummyTrainer()

    def tearDown(self):
        pass

    def test_is_scheduled(self):
        self.dummy_trainer.updates_per_epoch = 9
        self.dummy_trainer.update_number = 1
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 2))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 3))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 10))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 4))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 3))
        self.dummy_trainer.update_number = 2
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 2))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 3))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 10))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 3))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 2))
        self.dummy_trainer.update_number = 3
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 2))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 3))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 10))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 2))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 1))
        self.dummy_trainer.update_number = 4
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 2))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 3))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 10))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 1))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 0))
        self.dummy_trainer.update_number = 5
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 2))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 3))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 10))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 1))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 0))
        self.dummy_trainer.update_number = 6
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 2))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 3))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 10))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 3))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 2))
        self.dummy_trainer.update_number = 7
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 2))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 3))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 10))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 2))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 1))
        self.dummy_trainer.update_number = 8
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 2))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 3))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 10))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 1))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 0))
        self.dummy_trainer.update_number = 9
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 2))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 3))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 10))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 1))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 0))
        self.dummy_trainer.update_number = 10
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 2))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 3))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 10))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 4))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 3))
        self.dummy_trainer.update_number = 11
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 2))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 3))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 10))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 3))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 2))
        self.dummy_trainer.update_number = 12
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 2))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 3))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 10))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 2))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 2, 1))

        self.dummy_trainer.updates_per_epoch = 8
        self.dummy_trainer.update_number = 1
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 2))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 1))
        self.dummy_trainer.update_number = 2
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 1))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 0))
        self.dummy_trainer.update_number = 3
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 1))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 0))
        self.dummy_trainer.update_number = 4
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 2))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 1))
        self.dummy_trainer.update_number = 5
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 1))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 0))
        self.dummy_trainer.update_number = 6
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 1))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 0))
        self.dummy_trainer.update_number = 7
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 1))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 0))
        self.dummy_trainer.update_number = 8
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 1))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 0))
        self.dummy_trainer.update_number = 9
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 2))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 1))
        self.dummy_trainer.update_number = 10
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 1))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 0))
        self.dummy_trainer.update_number = 11
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 1))
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 0))
        self.dummy_trainer.update_number = 12
        self.assertTrue(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 2))
        self.assertFalse(BasicTrainer._is_scheduled(self.dummy_trainer, 3, 1))

if __name__ == '__main__':
    unittest.main()
