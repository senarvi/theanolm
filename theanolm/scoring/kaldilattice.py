#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the KaldiLattice used by Kaldi lattice rescorer.
"""

from collections import namedtuple
import logging

from theanolm.backend import InputError
from theanolm.backend.probfunctions import logprob_type
from theanolm.scoring.lattice import Lattice

def read_kaldi_vocabulary(input_file):
    """Reads a word-to-ID mapping from a Kaldi vocabulary file.

    :type input_file: file object
    :param input_file: a Kaldi vocabulary file (words.txt)

    :rtype: dict
    :returns: a mapping from words to Kaldi word IDs
    """

    result = dict()
    for line in input_file:
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 2:
            raise InputError("Invalid Kaldi vocabulary file.")
        word = parts[0]
        word_id = int(parts[1])
        result[word] = word_id
    return result

class KaldiLattice(Lattice):
    """Kaldi Lattice

    A word lattice that can be read in Kaldi CompactLattice Format
    """

    def __init__(self, lattice_lines, id_to_word):
        """Reads a Kaldi lattice file.

        If ``lattice_lines`` is ``None``, creates an empty lattice (useful for
        testing).

        :type lattice_lines: list of strs
        :param lattice_lines: list of lines in Kaldi CompactLattice text format

        :type id_to_word: list
        :param id_to_word: mapping of word IDs to words
        """

        super().__init__()

        # No logarithm base conversion.
        self._log_scale = logprob_type(1.0)

        self._initial_node_id = None
        self._final_node_id = None
        final_node = self.Node(None)
        final_node.final = True
        self.nodes = []

        if lattice_lines is None:
            self._num_nodes = 0
            self._num_links = 0
            return

        self.utterance_id = lattice_lines[0].strip()
        for line in lattice_lines[1:]:
            line = line.strip()
            parts = line.split()
            if not parts:
                continue
            if len(parts) == 1:
                state_to = kaldi_word_id = None
                str_weight = ""
            elif len(parts) == 2:
                state_to = kaldi_word_id = None
                str_weight = parts[1]
            elif len(parts) == 4:
                state_to = int(parts[1])
                kaldi_word_id = int(parts[2])
                str_weight = parts[3]
            else:
                raise InputError("Invalid number of fields in lattice `{}´ "
                                 "line `{}´.".format(self.utterance_id, line))
            state_from = int(parts[0])

            weight_parts = str_weight.split(',')
            graph_logprob = logprob_type(weight_parts[0]) \
                            if len(weight_parts) > 0 and weight_parts[0] \
                            else logprob_type(0.0)
            graph_logprob = -graph_logprob * self._log_scale
            ac_logprob = logprob_type(weight_parts[1]) \
                         if len(weight_parts) > 1 and weight_parts[1] \
                         else logprob_type(0.0)
            ac_logprob = -ac_logprob * self._log_scale
            transitions = weight_parts[2] if len(weight_parts) > 2 else ""

            if self._initial_node_id is None:
                self._initial_node_id = state_from

            for id in range(len(self.nodes), state_from + 1):
                self.nodes.append(self.Node(id))
            if state_to is not None:
                for id in range(len(self.nodes), state_to + 1):
                    self.nodes.append(self.Node(id))
                link = self._add_link(self.nodes[state_from], self.nodes[state_to])
                link.word = id_to_word[kaldi_word_id]
                if link.word == "<eps>":
                    link.word = None
                elif link.word == "#0":
                    raise InputError("Lattice `{}´ contains backoff transitions. "
                                     "Fix with Kaldi commands."
                                     .format(self.utterance_id))
                elif (link.word == "<s>") or (link.word == "</s>"):
                    raise InputError("Lattice `{}´ contains traditional start "
                                     "and end of sentence symbols."
                                     .format(self.utterance_id))
            else:
                link = self._add_link(self.nodes[state_from], final_node)
                link.word = None

            link.ac_logprob = ac_logprob
            link.lm_logprob = graph_logprob
            link.transitions = transitions

        if self._initial_node_id is None:
            raise InputError("No links in lattice `{}´."
                             .format(self.utterance_id))
        self.initial_node = self.nodes[self._initial_node_id]

        final_node.id = len(self.nodes)
        self.nodes.append(final_node)
