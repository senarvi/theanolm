#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from theanolm.probfunctions import logprob_type
from theanolm.scoring.lattice import Lattice


class KaldiLattice(Lattice):
    """Kaldi Lattice

    A word lattice that can be read in Kaldi CompactLattice Format
    """

    def __init__(self, lattice_file, word_map):
        """Reads an SLF lattice file.

        If ``lattice_file`` is ``None``, creates an empty lattice (useful for
        testing).

        :type lattice_file: file object
        :param lattice_file: a file in Kaldi CompactLattice text format
        """

        super().__init__()

        # No log conversion by default. "None" means the lattice file uses
        # linear probabilities.
        self._log_scale = logprob_type(1.0)

        self._initial_node_id = None
        # self._final_node_ids = []

        self.nodes = None

        if lattice_file is None:
            self._num_nodes = 0
            self._num_links = 0
            return

        for line in lattice_file:
            parts = line.split()
            assert len(parts) in {1,2,4}
            state_from = int(parts[0])
            state_to = kaldi_word_id = None
            str_weight = ""
            if len(parts) < 4:
                str_weight = parts[1] if len(parts) > 1 else ""
            else:
                state_to = int(parts[1])
                kaldi_word_id = int(parts[2])
                str_weight = parts[3]

            weight_parts = str_weight.split(',')
            graph_logprob = -(logprob_type(weight_parts[0]) if len(weight_parts) > 0 and len(weight_parts[0]) > 0 else logprob_type(0.0)) * self._log_scale

            ac_logprob = -(logprob_type(weight_parts[1]) if len(weight_parts) > 1 and len(weight_parts[1]) > 0 else logprob_type(0.0)) * self._log_scale
            transitions = weight_parts[2] if len(weight_parts) > 2 else ""

            if self._initial_node_id is None:
                self._initial_node_id = state_from

            self._ensure_node_present(state_from)
            if state_to is not None:
                self._ensure_node_present(state_to)
                link = self._add_link(self.nodes[state_from], self.nodes[state_to])
                link.word = word_map[kaldi_word_id]
                link.ac_logprob = ac_logprob
                link.graph_logprob = graph_logprob
                link.transitions = transitions
            else:
                self.nodes[state_from].final = True
                self.nodes[state_from].ac_logprob = ac_logprob
                self.nodes[state_from].lm_logprob = None
                self.nodes[state_from].graph_logprob = graph_logprob
                self.nodes[state_from].transitions = transitions
                self.nodes[state_from].word = "</s>"
                self.nodes[state_from].end_node = None
                # self._final_node_ids.append(state_from)

        assert self._initial_node_id is not None
        self.initial_node = self.nodes[self._initial_node_id]

        # assert len(self._final_node_ids) > 0
        # self.final_nodes = [self.nodes[id] for id in self._final_node_ids]

    def _ensure_node_present(self, node_id):
        if self.nodes is None:
            self.nodes = []

        for id in range(len(self.nodes), node_id+1):
            self.nodes.append(self.Node(id))

