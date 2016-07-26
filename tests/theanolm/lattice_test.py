#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
from theanolm.scoring.lattice import Lattice
from theanolm.scoring.slflattice import SLFLattice

class TestLattice(unittest.TestCase):
    def setUp(self):
        script_path = os.path.dirname(os.path.realpath(__file__))
        self.lattice_path = os.path.join(script_path, 'lattice.slf')

    def tearDown(self):
        pass

    def test_split_slf_line(self):
        lattice = SLFLattice()
        fields = lattice._split_slf_line('name=value '
                                         'name="va lue" '
                                         'WORD=\\"QUOTE '
                                         "WORD='CAUSE")
        self.assertEqual(fields[0], 'name=value')
        self.assertEqual(fields[1], 'name=va lue')
        self.assertEqual(fields[2], 'WORD="QUOTE')
        self.assertEqual(fields[3], "WORD='CAUSE")

    def test_split_slf_field(self):
        lattice = SLFLattice()
        name, value = lattice._split_slf_field("name=va 'lue")
        self.assertEqual(name, 'name')
        self.assertEqual(value, "va 'lue")

    def test_read_slf_header(self):
        lattice = SLFLattice()
        lattice._read_slf_header(['UTTERANCE=utterance #123'])
        self.assertEqual(lattice._utterance_id, 'utterance #123')
        lattice._read_slf_header(['U=utterance #456'])
        self.assertEqual(lattice._utterance_id, 'utterance #456')
        lattice._read_slf_header(['base=1.1', 'lmscale=1.2', 'wdpenalty=1.3'])
        self.assertEqual(lattice._log_base, 1.1)
        self.assertEqual(lattice._lm_scale, 1.2)
        self.assertEqual(lattice._wi_penalty, 1.3)
        lattice._read_slf_header(['start=2', 'end=3'])
        self.assertEqual(lattice._initial_node_id, 2)
        self.assertEqual(lattice._final_node_id, 3)
        lattice._read_slf_header(['NODES=5', 'LINKS=7'])
        self.assertEqual(lattice._num_nodes, 5)
        self.assertEqual(lattice._num_links, 7)
        lattice._read_slf_header(['N=8', 'L=9'])
        self.assertEqual(lattice._num_nodes, 8)
        self.assertEqual(lattice._num_links, 9)

    def test_read_slf_node(self):
        lattice = SLFLattice()
        lattice.nodes = [Lattice.Node(id) for id in range(5)]
        lattice._read_slf_node(0, [])
        lattice._read_slf_node(1, ['t=1.0'])
        lattice._read_slf_node(2, ['time=2.1'])
        lattice._read_slf_node(3, ['t=3.0', 'WORD=wo rd'])
        lattice._read_slf_node(4, ['time=4.1', 'W=word'])

        self.assertEqual(lattice.nodes[1].time, 1.0)
        self.assertEqual(lattice.nodes[2].time, 2.1)
        self.assertEqual(lattice.nodes[3].time, 3.0)
        self.assertEqual(lattice.nodes[3].word, 'wo rd')
        self.assertEqual(lattice.nodes[4].time, 4.1)
        self.assertEqual(lattice.nodes[4].word, 'word')

    def test_read_slf_link(self):
        lattice = SLFLattice()
        lattice.nodes = [Lattice.Node(id) for id in range(4)]
        lattice.links = []
        lattice._read_slf_node(0, ['t=0.0'])
        lattice._read_slf_node(1, ['t=1.0'])
        lattice._read_slf_node(2, ['t=2.0'])
        lattice._read_slf_node(3, ['t=3.0'])
        lattice._read_slf_link(0, ['START=0', 'END=1'])
        lattice._read_slf_link(1, ['S=1', 'E=2', 'WORD=wo rd', 'acoustic=-0.1', 'language=-0.2'])
        lattice._read_slf_link(2, ['S=2', 'E=3', 'W=word', 'a=-0.3', 'l=-0.4'])
        lattice._read_slf_link(3, ['S=1', 'E=3', 'a=-0.5', 'l=-0.6'])

        self.assertTrue(lattice.links[0].start_node is lattice.nodes[0])
        self.assertTrue(lattice.links[0].end_node is lattice.nodes[1])
        self.assertTrue(lattice.links[1].start_node is lattice.nodes[1])
        self.assertTrue(lattice.links[1].end_node is lattice.nodes[2])
        self.assertEqual(lattice.links[1].word, 'wo rd')
        self.assertEqual(lattice.links[1].ac_score, -0.1)
        self.assertEqual(lattice.links[1].lm_score, -0.2)
        self.assertTrue(lattice.links[2].start_node is lattice.nodes[2])
        self.assertTrue(lattice.links[2].end_node is lattice.nodes[3])
        self.assertEqual(lattice.links[2].word, 'word')
        self.assertEqual(lattice.links[2].ac_score, -0.3)
        self.assertEqual(lattice.links[2].lm_score, -0.4)
        self.assertTrue(lattice.links[3].start_node is lattice.nodes[1])
        self.assertTrue(lattice.links[3].end_node is lattice.nodes[3])
        self.assertEqual(lattice.links[3].ac_score, -0.5)
        self.assertEqual(lattice.links[3].lm_score, -0.6)

        self.assertEqual(len(lattice.nodes[0].in_links), 0)
        self.assertEqual(len(lattice.nodes[0].out_links), 1)
        self.assertEqual(len(lattice.nodes[1].in_links), 1)
        self.assertEqual(len(lattice.nodes[1].out_links), 2)
        self.assertEqual(len(lattice.nodes[2].in_links), 1)
        self.assertEqual(len(lattice.nodes[2].out_links), 1)
        self.assertEqual(len(lattice.nodes[3].in_links), 2)
        self.assertEqual(len(lattice.nodes[3].out_links), 0)

        self.assertEqual(lattice.nodes[0].out_links[0].end_node.time, 1.0)
        self.assertEqual(lattice.nodes[1].in_links[0].start_node.time, 0.0)
        self.assertEqual(lattice.nodes[1].out_links[0].end_node.time, 2.0)
        self.assertEqual(lattice.nodes[1].out_links[1].end_node.time, 3.0)
        self.assertEqual(lattice.nodes[2].in_links[0].start_node.time, 1.0)
        self.assertEqual(lattice.nodes[2].out_links[0].end_node.time, 3.0)
        self.assertEqual(lattice.nodes[3].in_links[0].start_node.time, 2.0)
        self.assertEqual(lattice.nodes[3].in_links[1].start_node.time, 1.0)

    def test_move_words_to_links(self):
        lattice = SLFLattice()
        lattice.nodes = [Lattice.Node(id) for id in range(5)]
        lattice.nodes[0].word = 'A'
        lattice.nodes[1].word = 'B'
        lattice.nodes[2].word = 'C'
        lattice.nodes[3].word = 'D'
        lattice.nodes[4].word = 'E'
        lattice.initial_node = lattice.nodes[0]
        lattice.final_node = lattice.nodes[4]
        lattice._add_link(lattice.nodes[0], lattice.nodes[1])
        lattice._add_link(lattice.nodes[0], lattice.nodes[2])
        lattice._add_link(lattice.nodes[1], lattice.nodes[3])
        lattice._add_link(lattice.nodes[2], lattice.nodes[3])
        lattice._add_link(lattice.nodes[3], lattice.nodes[4])
        lattice._move_words_to_links()
        self.assertEqual(lattice.links[0].word, 'B')
        self.assertEqual(lattice.links[1].word, 'C')
        self.assertEqual(lattice.links[2].word, 'D')
        self.assertEqual(lattice.links[3].word, 'D')
        self.assertEqual(lattice.links[4].word, 'E')
        for node in lattice.nodes:
            self.assertFalse(hasattr(node, 'word'))

    def test_sorted_nodes(self):
        lattice = Lattice()
        lattice.nodes = [Lattice.Node(id) for id in range(9)]
        lattice.nodes[0].time = 0.0
        lattice.nodes[2].time = 1.0
        lattice.nodes[4].time = 2.0
        lattice.nodes[3].time = 3.0
        lattice.nodes[5].time = 4.0
        lattice.nodes[1].time = 4.0
        lattice.nodes[6].time = 5.0
        lattice.nodes[7].time = None
        lattice.nodes[8].time = -1.0
        lattice._add_link(lattice.nodes[0], lattice.nodes[2])
        lattice._add_link(lattice.nodes[0], lattice.nodes[4])
        lattice._add_link(lattice.nodes[2], lattice.nodes[3])
        lattice._add_link(lattice.nodes[4], lattice.nodes[3])
        lattice._add_link(lattice.nodes[2], lattice.nodes[5])
        lattice._add_link(lattice.nodes[3], lattice.nodes[5])
        lattice._add_link(lattice.nodes[5], lattice.nodes[1])
        lattice._add_link(lattice.nodes[5], lattice.nodes[6])
        lattice._add_link(lattice.nodes[5], lattice.nodes[7])
        lattice._add_link(lattice.nodes[1], lattice.nodes[8])
        lattice._add_link(lattice.nodes[6], lattice.nodes[8])
        lattice._add_link(lattice.nodes[7], lattice.nodes[8])
        lattice.initial_node = lattice.nodes[0]
        lattice.final_node = lattice.nodes[8]

        sorted_nodes = lattice.sorted_nodes()
        self.assertEqual(sorted_nodes[0].id, 0)
        self.assertEqual(sorted_nodes[1].id, 2)
        self.assertEqual(sorted_nodes[2].id, 4)
        self.assertEqual(sorted_nodes[3].id, 3)
        self.assertEqual(sorted_nodes[4].id, 5)
        # Topologically equal nodes will be sorted in ascending time. The nodes
        # that don't have time will go last.
        self.assertEqual(sorted_nodes[5].id, 1)
        self.assertEqual(sorted_nodes[6].id, 6)
        self.assertEqual(sorted_nodes[7].id, 7)
        self.assertEqual(sorted_nodes[8].id, 8)

        lattice = SLFLattice()
        with open(self.lattice_path, 'r') as lattice_file:
            lattice.read(lattice_file)

        def reachable(initial_node, node):
            result = False
            for link in initial_node.out_links:
                if link.end_node is node:
                    result = True
            return result

        sorted_nodes = lattice.sorted_nodes()
        for left_node, right_node in zip(sorted_nodes, sorted_nodes[1:]):
            if (not left_node.time is None) and (not right_node.time is None):
                self.assertLessEqual(left_node.time, right_node.time)
                self.assertFalse(reachable(right_node, left_node))

    def test_read(self):
        lattice = SLFLattice()
        with open(self.lattice_path, 'r') as lattice_file:
            lattice.read(lattice_file)
        self.assertEqual(len(lattice.nodes), 24)
        self.assertEqual(len(lattice.links), 39)

if __name__ == '__main__':
    unittest.main()
