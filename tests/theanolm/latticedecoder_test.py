#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import math
from numpy.testing import assert_almost_equal
from theanolm.scoring import LatticeDecoder

class TestLatticeDecoder(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_compute_total_logprob(self):
        token = LatticeDecoder.Token(ac_logprob=math.log(0.1),
                                     lat_lm_logprob=math.log(0.2),
                                     nn_lm_logprob=math.log(0.3))
        token.compute_total_logprob(0.25)
        assert_almost_equal(token.total_logprob, math.log(0.1 * (0.25 * 0.3 + 0.75 * 0.2)))
        token.compute_total_logprob(0.25, 10.0)
        assert_almost_equal(token.total_logprob, math.log(0.1) + math.log(0.25 * 0.3 + 0.75 * 0.2) * 10.0)
        token = LatticeDecoder.Token(ac_logprob=-1000,
                                     lat_lm_logprob=-1001,
                                     nn_lm_logprob=-1002)
        token.compute_total_logprob(0.75)
        # ln(exp(-1000) * (0.75 * exp(-1002) + 0.25 * exp(-1001)))
        assert_almost_equal(token.total_logprob, -2001.64263, decimal=4)

if __name__ == '__main__':
    unittest.main()
