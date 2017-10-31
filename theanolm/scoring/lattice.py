#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the Lattice class used by word lattice decoders.
"""

import logging

from theanolm.backend import InputError

class NodeNotFoundError(Exception):
    pass

class Lattice(object):
    """Word Lattice

    Word lattice describes a search space for decoding. The graph is represented
    as a list of nodes and links. Each node contains pointers to its incoming
    and outgoing links. Each link contains a pointer to the nodes in both ends.
    """

    class Link(object):
        """A link between two graph nodes.

        A link contains pointers to the start and end node. A node that has the
        link as an outgoing link can find the next node from ``end_node`` and a
        node that has the link as an incoming link can find the previous node
        from ``start_node``.
        """

        def __init__(self, start_node, end_node, word=None,
                     ac_logprob=None, lm_logprob=None, transitions=""):
            """Constructs a link.

            :type start_node: self.Node
            :param start_node: the node that has this link as an outgoing link

            :type end_node: self.Node
            :param end_node: the node that has this link as an incoming link

            :type word: str or int
            :param word: the word label on the link

            :type ac_logprob: float
            :param ac_logprob: acoustic log probability

            :type lm_logprob: float
            :param lm_logprob: language model log probability

            :type transitions: str
            :param transitions: transitions for an FST lattice
            """

            self.start_node = start_node
            self.end_node = end_node
            self.word = None
            self.ac_logprob = ac_logprob
            self.lm_logprob = lm_logprob
            self.transitions = transitions

    class Node(object):
        """A node in the graph.

        Outgoing and incoming links can be used to find the next and previous
        nodes in the topology.
        """

        def __init__(self, node_id):
            """Constructs a node with no links.

            :type node_id: int
            :param node_id: the ID that can be used to access the node in the
                            node list
            """

            self.id = node_id
            self.out_links = []
            self.in_links = []
            self.time = None
            self.best_logprob = None
            self.final = False

    def __init__(self):
        """Constructs an empty lattice.
        """

        self.nodes = []
        self.links = []
        self.initial_node = None
        self.utterance_id = None
        self.lm_scale = None
        self.wi_penalty = None

    @classmethod
    def from_decoder(cls, original_lattice, final_tokens, recomb_tokens,
                     vocabulary):
        """Constructs a lattice from the paths visited when decoding a lattice.

        :type original_lattice: Lattice
        :param original_lattice: the original lattice that was decoded; the new
            lattice will be based on the links found in this lattice

        :type final_tokens: list of LatticeDecoder.Tokens
        :param final_tokens: the tokens that reached the final node in the
            decoder; the resulting lattice will always contain these paths

        :type recomb_tokens: list of tuples
        :param recomb_tokens: tokens that were dropped during recombination;
            each tuple contains the token, and the word IDs and NNLM log
            probability of the token that was kept when dropping the token

        :type vocabulary: Vocabulary
        :param vocabulary: vocabulary for mapping the word IDs in the tokens to
            the words in the lattices

        :rtype: Lattice
        :returns: the created lattice
        """

        def get_node(word_ids, create=True):
            words = [vocabulary.id_to_word[id] if isinstance(id, int) else id
                     for id in word_ids[1:]]
            return self._reconstruct_path(words, original_lattice, create)

        self = cls()
        self.initial_node = self.Node(0)
        self.initial_node.word_to_link = dict()
        self.final_node = self.Node(1)
        self.final_node.word_to_link = dict()
        self.nodes = [self.initial_node, self.final_node]

        # Add a mapping from words to outgoing links on each node to speed up
        # the process.
        for node in original_lattice.nodes:
            node.word_to_link = {link.word: link for link in node.out_links}
            assert len(node.word_to_link) == len(node.out_links)

        # Create all the paths that correspond to the final tokens.
        for token in final_tokens:
            node = get_node(token.history)
            # XXX Should we have the logprob of the final link instead of the
            # token here?
            link = self.Link(node, self.final_node, word=None,
                             ac_logprob=0,
                             lm_logprob=token.nn_lm_logprob,
                             transitions="")
            node.out_links.append(link)
            self.links.append(link)

        # Add recombined tokens.
        for token, new_history, nn_lm_logprob in reversed(recomb_tokens):
            assert new_history[-1] == token.history[-1]
            word_id = token.history[-1]
            if isinstance(word_id, int):
                word = vocabulary.id_to_word[word_id]
            else:
                word = word_id
                logging.debug("Out-of-vocabulary word in lattice: %s", word)

            # Find the incoming link from the token that was kept during
            # recombination.
            try:
                recomb_from_node = get_node(new_history[:-1], False)
            except NodeNotFoundError:
                continue
            if word not in recomb_from_node.word_to_link:
                continue
            recomb_link = recomb_from_node.word_to_link[word]

            # Add a link from the previous node in the token history to the node
            # that was kept during recombination. The difference in LM log
            # probability can be computed from the token (path) NNLM log
            # probabilities.
            from_node = get_node(token.history[:-1])
            lm_diff = token.nn_lm_logprob - nn_lm_logprob
            new_link = self.Link(from_node, recomb_link.end_node, word
                                 recomb_link.ac_logprob,
                                 recomb_link.lm_logprob + lm_diff,
                                 recomb_link.transitions)
            from_node.out_links.append(new_link)
            from_node.word_to_link[word] = new_link
            self.links.append(new_link)

        return self

    def write_kaldi(self, utterance_id, output_file, word_to_id):
        """Writes the lattice in Kaldi CompactLattice format.

        :type utterance_id: str
        :param utterance_id: name of the lattice

        :type output_file: file object
        :param output_file: a file where to write the output

        :type word_to_id: Vocabulary
        :param word_to_id: mapping of words to Kaldi IDs
        """

        def write_normal_link(link):
            word = link.word
            if word == "</s>":
                word = "!SENT_END"
            word_id = word_to_id[word]
            output_file.write("{} {} {} {},{},{}\n".format(
                link.start_node.id,
                link.end_node.id,
                word_id,
                -link.lm_logprob,
                -link.ac_logprob,
                link.transitions))

        def write_final_link(link):
            output_file.write("{} {},{},{}\n".format(
                link.start_node.id,
                -link.lm_logprob,
                -link.ac_logprob,
                link.transitions))

        word_to_id['!SENT_END'] = 0

        output_file.write("{}\n".format(utterance_id))
        for node_id, node in enumerate(self.nodes):
            for link in node.out_links:
                if link.word is None:
                    write_final_link(link)
                else:
                    write_normal_link(link)
        output_file.write("\n")

    def sorted_nodes(self):
        """Sorts nodes topologically, then by time.

        Returns a list which contains the nodes in sorted order. Uses the Kahn's
        algorithm to sort the nodes topologically, but always picks the node
        from the queue that has the lowest time stamp, if the nodes contain time
        stamps.
        """

        result = []
        # A queue of nodes to be visited next:
        node_queue = [self.initial_node]
        # The number of incoming links not traversed yet:
        in_degrees = [len(node.in_links) for node in self.nodes]
        while node_queue:
            node = node_queue.pop()
            result.append(node)
            for link in node.out_links:
                next_node = link.end_node
                in_degrees[next_node.id] -= 1
                if in_degrees[next_node.id] == 0:
                    node_queue.append(next_node)
                    node_queue.sort(key=lambda x: (x.time is None, x.time),
                                    reverse=True)
                elif in_degrees[next_node.id] < 0:
                    raise InputError("Word lattice contains a cycle.")

        if len(result) < len(self.nodes):
            logging.warning("Word lattice contains unreachable nodes.")
        else:
            assert len(result) == len(self.nodes)

        return result

    def _add_link(self, start_node, end_node):
        """Adds a link between two nodes.

        :type start_node: Node
        :param start_node: creates a link from this node

        :type end_node: Node
        :param end_node: creates a link to this node

        :rtype: Link
        :returns: the created link
        """

        link = self.Link(start_node, end_node)
        self.links.append(link)
        start_node.out_links.append(link)
        end_node.in_links.append(link)
        return link

    def _reconstruct_path(self, words, original_lattice, create=True):
        """Ensures that a path exists, creating new nodes if necessary, and
        returns the last node of the path.

        :type words: list of strs
        :param words: words of the path

        :type original_lattice: Lattice
        :param original_lattice: the original lattice where the probabilities of
            the links will be read from

        :type create: bool
        :param create: if set to ``False``, won't create any nodes but raises a
            ``NodeNotFoundError`` if the path doesn't exist

        :rtype: Lattice.Node
        :returns: the last node of the path
        """

        node = self.initial_node
        orig_node = original_lattice.initial_node

        for word in words:
            if word not in orig_node.word_to_link:
                continue
            orig_link = orig_node.word_to_link[word]

            if word in node.word_to_link:
                link = node.word_to_link[word]
            else:
                if not create:
                    raise NodeNotFoundError
                end_node_id = len(self.nodes)
                end_node = self.Node(end_node_id)
                end_node.word_to_link = dict()
                self.nodes.append(end_node)
                link = self.Link(node, end_node, word,
                                 orig_link.ac_logprob,
                                 orig_link.lm_logprob,
                                 orig_link.transitions)
                node.out_links.append(link)
                node.word_to_link[word] = link
                self.links.append(link)

            node = link.end_node
            orig_node = orig_link.end_node

        return node
