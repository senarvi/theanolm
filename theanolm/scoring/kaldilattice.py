#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the KaldiLattice used by Kaldi lattice rescorer.
"""

from collections import namedtuple
import logging

from theanolm.backend.probfunctions import logprob_type
from theanolm.scoring.lattice import Lattice

class KaldiLattice(Lattice):
    """Kaldi Lattice

    A word lattice that can be read in Kaldi CompactLattice Format
    """

    def __init__(self, lattice_file, word_map):
        """Reads a Kaldi lattice file.

        If ``lattice_file`` is ``None``, creates an empty lattice (useful for
        testing).

        :type lattice_file: file object
        :param lattice_file: a file in Kaldi CompactLattice text format

        :type word_map: list
        :param word_map: mapping of word IDs to words
        """

        super().__init__()

        # No log conversion by default. "None" means the lattice file uses
        # linear probabilities.
        self._log_scale = logprob_type(1.0)

        self._initial_node_id = None
        self._final_node_id = None
        # self._final_node_ids = []
        final_node = self.Node(2**32)
        final_node.final = True

        self.nodes = None

        if lattice_file is None:
            self._num_nodes = 0
            self._num_links = 0
            return

        for line in lattice_file:
            parts = line.split()
            if len(parts) == 1:
                state_to = kaldi_word_id = None
                str_weight = ""
            elif len(parts) == 2:
                state_to = kaldi_word_id = None
                str_weight = parts[1]
            elif len(parts) == 4:
                state_to = int(parts[1])
                kaldi_word_id = int(parts[2])
                if kaldi_word_id == 0:
                    raise InputError("Zero word ID in a CompactLattice file.")
                str_weight = parts[3]
            else:
                raise InputError("Invalid number of fields in a CompactLattice file.")
            state_from = int(parts[0])

            weight_parts = str_weight.split(',')
            graph_logprob = logprob_type(weight_parts[0])
                            if len(weight_parts) > 0 and weight_parts[0]
                            else logprob_type(0.0)
            graph_logprob = -graph_logprob * self._log_scale
            ac_logprob = logprob_type(weight_parts[1])
                         if len(weight_parts) > 1 and weight_parts[1]
                         else logprob_type(0.0)
            ac_logprob = -ac_logprob * self._log_scale
            transitions = weight_parts[2] if len(weight_parts) > 2 else ""

            if self._initial_node_id is None:
                self._initial_node_id = state_from

            self._ensure_node_present(state_from)
            if state_to is not None:
                self._ensure_node_present(state_to)
                link = self._add_link(self.nodes[state_from], self.nodes[state_to])
                link.word = word_map[kaldi_word_id]
                if link.word == "#0":
                    raise InputError("Lattice contains backoff transitions. "
                                     "Fix with Kaldi commands.")
                assert link.word != "<s>"
                assert link.word != "</s>"
            else:
                link = self._add_link(self.nodes[state_from], final_node)
                link.word = "!SENT_END"
                #assert transitions == ""

            link.ac_logprob = ac_logprob
            link.lm_logprob = graph_logprob
            link.transitions = transitions

        final_node.id = len(self.nodes)
        self.nodes.append(final_node)

        assert self._initial_node_id is not None
        self.initial_node = self.nodes[self._initial_node_id]

        # assert len(self._final_node_ids) > 0
        # self.final_nodes = [self.nodes[id] for id in self._final_node_ids]

    def _ensure_node_present(self, node_id):
        if self.nodes is None:
            self.nodes = []

        for id in range(len(self.nodes), node_id+1):
            self.nodes.append(self.Node(id))
