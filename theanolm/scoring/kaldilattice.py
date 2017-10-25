#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import namedtuple

import logging

from theanolm.backend.probfunctions import logprob_type
from theanolm.scoring.lattice import Lattice


class KaldiLattice(Lattice):
    """Kaldi Lattice

    A word lattice that can be read in Kaldi CompactLattice Format
    """

    def __init__(self, lattice_file, word_map):
        """Reads an Kaldi lattice file.

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
            assert len(parts) in {1,2,4}
            state_from = int(parts[0])
            state_to = kaldi_word_id = None
            str_weight = ""
            if len(parts) < 4:
                str_weight = parts[1] if len(parts) > 1 else ""
            else:
                state_to = int(parts[1])
                kaldi_word_id = int(parts[2])
                assert kaldi_word_id != 0
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
                assert link.word != "#0" # Don't want backoff transitions in this lattice, fix with kaldi commands if this is the case
                assert link.word != "<s>"
                assert link.word != "</s>"
            else:
                link = self._add_link(self.nodes[state_from], final_node)
                link.word = "!SENT_END"
                #assert transitions == ""

            link.ac_logprob = ac_logprob
            link.graph_logprob = graph_logprob
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

class NodeNotFoundError(Exception):
    pass

class OutKaldiLattice(object):

    edge = namedtuple('Edge', ['target', 'lm_weight', 'am_weight', 'transitions'])

    class Node(object):
        def __init__(self):
            self.out = {}
            self.final = None

    def __init__(self):
        self._nodes = [self.Node()]

    def _new_node(self):
        self._nodes.append(self.Node())
        return len(self._nodes) - 1

    def _reconstruct_path(self, kaldi_path, word_path, orig_lat, create=True):
        kaldi_node_id = 0
        kaldi_node = self._nodes[kaldi_node_id]
        orig_node = orig_lat.initial_node

        for k, w in zip(kaldi_path, word_path):
            if w not in orig_node.out_links_map:
                pass
            link = orig_node.out_links_map[w]

            if k not in kaldi_node.out:
                if not create:
                    raise NodeNotFoundError
                new_node_id = self._new_node()
                kaldi_node.out[k] = self.edge(new_node_id, -link.graph_logprob, -link.ac_logprob, link.transitions)

            kaldi_node_id = kaldi_node.out[k].target
            kaldi_node = self._nodes[kaldi_node_id]
            orig_node = link.end_node

        return kaldi_node_id, kaldi_node


    def create_network(self, orig_lat, final_tokens, recomb_tokens, token_vocabulary, kaldi_vocabulary):
        kaldi_vocabulary['!SENT_END'] = 0


        def get_node(token_hist, create=True):
            word_path = [token_vocabulary.id_to_word[i] if type(i) == int else i for i in token_hist[1:]]
            word_path = ["!SENT_END" if w == "</s>" else w for w in word_path]
            kaldi_path = [kaldi_vocabulary[w] for w in word_path]
            return self._reconstruct_path(kaldi_path, word_path, orig_lat, create)

        for node in orig_lat.nodes:
            node.out_links_map = {l.word: l for l in node.out_links}
            assert len(node.out_links_map) == len(node.out_links)

        for token in final_tokens:
            _, node = get_node(token.history)
            node.final = self.edge(0, -token.nn_lm_logprob, 0, "")

        for token, new_history, nnlm_prob in reversed(recomb_tokens):
            assert new_history[-1] == token.history[-1]
            try:
                _, recomb_from_node = get_node(new_history[:-1], False)
            except NodeNotFoundError:
                continue
            label = token.history[-1]
            if type(label) == int:
                label = kaldi_vocabulary[token_vocabulary.id_to_word[token.history[-1]]]
            else:
                logging.debug("Interesting, label: {}".format(label))
            if label not in recomb_from_node.out:
                continue

            _, from_node = get_node(token.history[:-1])

            from_node.out[label] = recomb_from_node.out[label]._replace(lm_weight=recomb_from_node.out[label].lm_weight + (nnlm_prob - token.nn_lm_logprob))


    def write(self, key, out):
        out.write("{}\n".format(key))
        for i, node in enumerate(self._nodes):
            for label, edge in node.out.items():
                out.write("{} {} {} {},{},{}\n".format(
                    i,
                    edge.target,
                    label,
                    edge.lm_weight,
                    edge.am_weight,
                    edge.transitions
                ))
            if node.final is not None:
                out.write("{} {},{},{}\n".format(
                    i,
                    node.final.lm_weight,
                    node.final.am_weight,
                    node.final.transitions
                ))
        out.write("\n")