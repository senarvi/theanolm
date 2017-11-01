#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the Lattice class used by word lattice decoders.
"""

import logging

from theanolm.backend import InputError
from theanolm.scoring.lattice import Lattice

class NodeNotFoundError(Exception):
    pass

class RescoredLattice(Lattice):
    """Rescored Lattice

    Word lattice that is created from the tokens after decoding another lattice.
    The LM probabilities in the lattice are replaced with interpolated
    probabilities from the TheanoLM model and those in the original lattice.
    """

    def __init__(self, original_lattice, final_tokens, recomb_tokens,
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

        self.utterance_id = original_lattice.utterance_id

        self.initial_node = self.Node(0)
        self.initial_node.word_to_link = dict()
        self.nodes = [self.initial_node]
        final_node = self.Node(None)
        final_node.final = True
        final_node.word_to_link = dict()

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
            link = self.Link(node, final_node, word=None,
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
            new_link = self.Link(from_node, recomb_link.end_node, word,
                                 recomb_link.ac_logprob,
                                 recomb_link.lm_logprob + lm_diff,
                                 recomb_link.transitions)
            from_node.out_links.append(new_link)
            from_node.word_to_link[word] = new_link
            self.links.append(new_link)

        final_node.id = len(self.nodes)
        self.nodes.append(final_node)
        return self

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
