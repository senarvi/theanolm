#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the Lattice class used by word lattice decoders.
"""

import logging

from theanolm.backend import InputError
from theanolm.scoring.lattice import Lattice

class NodeNotFoundError(Exception):
    pass

def _follow_word_from_node(node, word):
    """Follows the link with given word label from given node.

    If there is a link from ``node`` with the label ``word``, returns the end
    node and the log probabilities and transition IDs of the link. If there are
    null links in between, returns the sum of the log probabilities and the
    concatenation of the transition IDs.

    :type node: Lattice.Node
    :param node: node where to start searching

    :type word: str
    :param word: word to search for

    :rtype: tuple of (Lattice.Node, float, float, str)
    :returns: the end node of the link with the word label (or ``None`` if the
              word is not found), and the total acoustic log probability, LM log
              probability, and transition IDs of the path to the word
    """

    if word not in node.word_to_link:
        return (None, None, None, None)

    link = node.word_to_link[word]
    if link.word is not None:
        return (link.end_node,
                link.ac_logprob if link.ac_logprob is not None else 0.0,
                link.lm_logprob if link.lm_logprob is not None else 0.0,
                link.transitions if link.transitions is not None else "")

    end_node, ac_logprob, lm_logprob, transitions = \
        _follow_word_from_node(link.end_node, word)
    if end_node is None:
        return (None, None, None, None)
    else:
        if link.ac_logprob is not None:
            ac_logprob += link.ac_logprob
        if link.lm_logprob is not None:
            lm_logprob += link.lm_logprob
        if link.transitions is not None:
            transitions += link.transitions
        return (end_node, ac_logprob, lm_logprob, transitions)

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

        def follow_word_ids(word_ids, create=True):
            """Follows a path from the initial node with given word IDs.

            :type word_ids: list of ints
            :param word_ids: IDs of the words to be found on the path

            :type create: bool
            :param create: if ``True``, creates new nodes if necessary

            :rtype: Lattice.Node
            :returns: the last node of the path
            """

            words = [vocabulary.id_to_word[id] if isinstance(id, int) else id
                     for id in word_ids[1:]]
            return self._follow_words(words, original_lattice, create)

        super().__init__()

        self.utterance_id = original_lattice.utterance_id

        self.initial_node = self.Node(len(self.nodes))
        self.initial_node.word_to_link = dict()
        self.nodes.append(self.initial_node)
        final_node = self.Node(None)
        final_node.final = True
        final_node.word_to_link = dict()

        self._add_word_maps(original_lattice.nodes)

        # Create all the paths that correspond to the final tokens.
        for token in final_tokens:
            node = follow_word_ids(token.history)
            # If the lattice is not determinized, it may happen that we have two
            # tokens with the same word sequence.
            if any(link.end_node.final for link in node.out_links):
                continue
            # For the final link we set the NNLM log probability of the entire
            # path. The created lattice is a tree, so we can equivalently set
            # the entire path probability in the final link.
            link = self.Link(node, final_node, word=None,
                             ac_logprob=0,
                             lm_logprob=token.nn_lm_logprob,
                             transitions="")
            node.out_links.append(link)
            self.links.append(link)

        # Add recombined tokens.
        oov_words = set()
        for token, new_history, nn_lm_logprob in reversed(recomb_tokens):
            assert new_history[-1] == token.history[-1]
            word_id = token.history[-1]
            if isinstance(word_id, int):
                word = vocabulary.id_to_word[word_id]
            else:
                word = word_id
                oov_words.add(word)

            # Find the incoming link that corresponds to the token that was kept
            # during recombination.
            try:
                recomb_from_node = follow_word_ids(new_history[:-1], False)
            except NodeNotFoundError:
                continue
            # Our new lattice doesn't contain null links, so word_to_link maps
            # never skip nodes.
            if word not in recomb_from_node.word_to_link:
                continue
            recomb_link = recomb_from_node.word_to_link[word]

            # Add a link from the previous word in the token history to the node
            # that was kept during recombination. The difference in LM log
            # probability can be computed from the token (path) NNLM log
            # probabilities.
            from_node = follow_word_ids(token.history[:-1])
            lm_logprob_diff = token.nn_lm_logprob - nn_lm_logprob
            new_link = self.Link(from_node, recomb_link.end_node, word,
                                 recomb_link.ac_logprob,
                                 recomb_link.lm_logprob + lm_logprob_diff,
                                 recomb_link.transitions)
            from_node.out_links.append(new_link)
            # Tokens never contain null words, so we can be sure that
            # word_to_link maps in our new lattice never skip nodes.
            assert word is not None
            from_node.word_to_link[word] = new_link
            self.links.append(new_link)

        if oov_words:
            logging.debug("Out-of-vocabulary words in lattice: %s",
                          ', '.join(oov_words))

        final_node.id = len(self.nodes)
        self.nodes.append(final_node)

    def _add_word_maps(self, nodes):
        """Adds mapping from words to outgoing links on each node.

        For each node, finds the words that follow the node. If an outgoing link
        is a null link, follows the link recursively. Then adds the attribute
        ``word_to_link`` to the node that provides a mapping from words to the
        links that have to be followed to reach that word. This is done to avoid
        doing the search every time when constructing the rescored lattice.

        :type nodes: list of Lattice.Nodes
        :param nodes: add a ``word_to_link`` map to these nodes
        """

        def next_words(link):
            if link.word is None:
                return [word
                        for next_link in link.end_node.out_links
                        for word in next_words(next_link)]
            else:
                return [link.word]

        determinized = True
        for node in nodes:
            node.word_to_link = {word: link
                                 for link in node.out_links
                                 for word in next_words(link)}
            if len(node.word_to_link) != len(node.out_links):
                determinized = False
        if not determinized:
            logging.warning(
                "The original word lattice is not determinized, meaning that "
                "there are multiple links from the same node with the same "
                "word label, or the lattice contains null links (epsilon "
                "arcs). Lattice rescoring may not work properly.")

    def _follow_words(self, words, original_lattice, create=True):
        """Ensures that a path exists, creating new nodes if necessary, and
        returns the last node of the path.

        The new lattice will be created by repeatedly calling this function. It
        starts from the beginning of the lattice and follows the nodes that
        correspond to the given words. When a word is not found from the
        outgoing links of the node, the tail will be created. Thus the created
        lattice is always a tree.

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
            # The original lattice may contain null links that we have to skip.
            orig_end_node, ac_logprob, lm_logprob, transitions = \
                _follow_word_from_node(orig_node, word)
            if orig_end_node is None:
                continue

            # Our new lattice doesn't contain null links, so word_to_link maps
            # never skip nodes.
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
                                 ac_logprob, lm_logprob, transitions)
                node.out_links.append(link)
                node.word_to_link[word] = link
                self.links.append(link)

            node = link.end_node
            orig_node = orig_end_node

        return node
