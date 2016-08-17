#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
import math
import logging
import numpy
import theano
from theano import tensor
from theanolm.network import RecurrentState
from theanolm.probfunctions import *

class LatticeDecoder(object):
    """Word Lattice Decoding Using a Neural Network Language Model
    """

    class Token:
        """Decoding Token

        A token represents a partial path through a word lattice. The decoder
        propagates a set of tokens through the lattice by
        """

        def __init__(self,
                     history=[],
                     state=[],
                     ac_logprob=logprob_type(0.0),
                     lat_lm_logprob=logprob_type(0.0),
                     nn_lm_logprob=logprob_type(0.0)):
            """Constructs a token with given recurrent state and logprobs.

            The constructor won't compute the total logprob. The user is
            responsible for computing it when necessary, to avoid unnecessary
            overhead.

            New tokens will not have recombination hash and total log
            probability set.

            :type history: list of ints
            :param history: word IDs that the token has passed

            :type state: RecurrentState
            :param state: the state of the recurrent layers for a single
                          sequence

            :type ac_logprob: logprob_type
            :param ac_logprob: sum of the acoustic log probabilities of the
                               lattice links

            :type lat_lm_logprob: logprob_type
            :param lat_lm_logprob: sum of the LM log probabilities of the
                                   lattice links

            :type nn_lm_logprob: logprob_type
            :param nn_lm_logprob: sum of the NNLM log probabilities of the
                                  lattice links
            """

            self.history = history
            self.state = state
            self.ac_logprob = ac_logprob
            self.lat_lm_logprob = lat_lm_logprob
            self.nn_lm_logprob = nn_lm_logprob
            self.recombination_hash = None
            self.total_logprob = None

        @classmethod
        def copy(classname, token):
            """Creates a copy of a token.

            The recurrent layer states will not be copied - a pointer will be
            copied instead. There's no need to copy the structure, since we
            never modify the state of a token, but replace it if necessary.

            Recombination hash and total log probability will not be copied.

            :type token: LatticeDecoder.Token
            :param token: a token to copy

            :rtype: LatticeDecoder.Token
            :returns: a copy of ``token``
            """

            return classname(deepcopy(token.history),
                             token.state,
                             token.ac_logprob,
                             token.lat_lm_logprob,
                             token.nn_lm_logprob)

        def recompute_hash(self, recombination_order):
            """Computes the hash that will be used to decide if two tokens
            should be recombined.

            :type recombination_order: int
            :param recombination_order: number of words to consider when
                recombining tokens, or ``None`` for the entire history
            """

            if recombination_order is None:
                limited_history = self.history
            else:
                limited_history = self.history[-recombination_order:]
            self.recombination_hash = hash(tuple(limited_history))

        def recompute_total(self, nn_lm_weight, lm_scale, wi_penalty,
                            linear=False):
            """Computes the interpolated language model log probability and
            the total log probability.

            The interpolated LM log probability is saved in ``self.lm_logprob``.
            The total log probability is computed by applying LM scale factor
            and adding the acoustic log probability and word insertion penalty.

            :type nn_lm_weight: logprob_type
            :param nn_lm_weight: weight of the neural network LM probability
                                 when interpolating with the lattice probability

            :type lm_scale: logprob_type
            :param lm_scale: scaling factor for LM probability when computing
                             the total probability

            :type wi_penalty: logprob_type
            :param wi_penalty: penalize each word in the history by adding this
                               value as many times as there are words

            :type linear: bool
            :param linear: if set to ``True`` performs linear interpolation
                           instead of (pseudo) log-linear
            """

            if linear:
                self.lm_logprob = interpolate_linear(
                    self.nn_lm_logprob, self.lat_lm_logprob,
                    nn_lm_weight)
            else:
                self.lm_logprob = interpolate_loglinear(
                    self.nn_lm_logprob, self.lat_lm_logprob,
                    nn_lm_weight, (1.0 - nn_lm_weight))
            self.total_logprob = self.ac_logprob
            self.total_logprob += self.lm_logprob * lm_scale
            self.total_logprob += wi_penalty * len(self.history)

        def __str__(self, vocabulary=None):
            """Creates a string representation of the token.

            :type vocabulary: Vocabulary
            :param vocabulary: if a vocabulary is given, uses it to decode
                               history word names from word IDs

            :rtype: str
            :returns: a string that includes all the attributes in one line
            """

            if vocabulary is None:
                history = ' '.join(str(x) for x in self.history)
            else:
                history = ' '.join(vocabulary.id_to_word[self.history])

            if self.total_logprob is None:
                return '[{}]  acoustic: {:.2f}  lattice LM: {:.2f}  NNLM: ' \
                       '{:.2f}'.format(
                           history,
                           self.ac_logprob,
                           self.lat_lm_logprob,
                           self.nn_lm_logprob)
            else:
                return '[{}]  acoustic: {:.2f}  lattice LM: {:.2f}  NNLM: ' \
                       '{:.2f}  total: {:.2f}'.format(
                           history,
                           self.ac_logprob,
                           self.lat_lm_logprob,
                           self.nn_lm_logprob,
                           self.total_logprob)

    def __init__(self, network, decoding_options, profile=False):
        """Creates a Theano function that computes the output probabilities for
        a single time step.

        Creates the function self.step_function that takes as input a set of
        word sequences and the current recurrent states. It uses the previous
        states and word IDs to compute the output distributions, and computes
        the probabilities of the target words.

        All invocations of ``decode()`` will use the given NNLM weight and LM
        scale when computing the total probability. If LM scale is not given,
        uses the value provided in the lattice files. If it's not provided in a
        lattice file either, performs no scaling of LM log probabilities.

        ``decoding_options`` should countain the following elements:

        nnlm_weight : float
          weight of the neural network probabilities when interpolating with the
          lattice probabilities

        lm_scale : float
          if other than ``None``, the decoder will scale language model log
          probabilities by this factor; otherwise the scaling factor will be
          read from the lattice file

        wi_penalty : float
          penalize word insertion by adding this value to the total log
          probability of a token as many times as there are words

        ignore_unk : bool
          if set to ``True``, <unk> tokens are excluded from perplexity
          computation

        unk_penalty : float
          if set to othern than None, used as <unk> token score

        linear_interpolation : bool
          if set to ``True``, use linear instead of (pseudo) log-linear
          interpolation of language model probabilities

        max_tokens_per_node : int
          if set to other than None, leave only this many tokens at each node

        beam : float
          if set to other than None, prune tokens whose total log probability is
          further than this from the best token at each point in time

        recombination_order : int
          number of words to consider when deciding whether two tokens should be
          recombined, or ``None`` for the entire word history

        :type network: Network
        :param network: the neural network object

        :type decoding_options: dict
        :param decoding_options: a dictionary of decoding options (see above)

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        self._network = network
        self._vocabulary = network.vocabulary
        self._nnlm_weight = logprob_type(decoding_options['nnlm_weight'])
        self._lm_scale = decoding_options['lm_scale']
        self._wi_penalty = decoding_options['wi_penalty']
        self._ignore_unk = decoding_options['ignore_unk']
        self._unk_penalty = decoding_options['unk_penalty']
        self._linear_interpolation = decoding_options['linear_interpolation']
        self._max_tokens_per_node = decoding_options['max_tokens_per_node']
        self._beam = decoding_options['beam']
        if not self._beam is None:
            self._beam = logprob_type(self._beam)
        self._recombination_order = decoding_options['recombination_order']

        self._sos_id = self._vocabulary.word_to_id['<s>']
        self._eos_id = self._vocabulary.word_to_id['</s>']
        self._unk_id = self._vocabulary.word_to_id['<unk>']

        inputs = [network.word_input,
                  network.class_input,
                  network.target_class_ids]
        inputs.extend(network.recurrent_state_input)

        outputs = [tensor.log(network.target_probs())]
        outputs.extend(network.recurrent_state_output)

        # Ignore unused input, because is_training is only used by dropout
        # layer.
        self.step_function = theano.function(
            inputs,
            outputs,
            givens=[(network.is_training, numpy.int8(0))],
            name='step_predictor',
            on_unused_input='ignore')

    def decode(self, lattice):
        """Propagates tokens through given lattice and returns a list of tokens
        in the final node.

        Propagates tokens at a node to every outgoing link by creating a copy of
        each token and updating the language model scores according to the link.

        :type lattice: Lattice
        :param lattice: a word lattice to be decoded

        :rtype: list of LatticeDecoder.Tokens
        :returns: the final tokens sorted by total log probability in descending
                  order
        """

        if not self._lm_scale is None:
            lm_scale = logprob_type(self._lm_scale)
        elif not lattice.lm_scale is None:
            lm_scale = logprob_type(lattice.lm_scale)
        else:
            lm_scale = logprob_type(1.0)

        if not self._wi_penalty is None:
            wi_penalty = logprob_type(self._wi_penalty)
        if not lattice.wi_penalty is None:
            wi_penalty = logprob_type(lattice.wi_penalty)
        else:
            wi_penalty = logprob_type(0.0)

        self._tokens = [list() for _ in lattice.nodes]
        initial_state = RecurrentState(self._network.recurrent_state_size)
        initial_token = self.Token(history=[self._sos_id], state=initial_state)
        initial_token.recompute_hash(self._recombination_order)
        initial_token.recompute_total(self._nnlm_weight, lm_scale, wi_penalty,
                                      self._linear_interpolation)
        self._tokens[lattice.initial_node.id].append(initial_token)
        lattice.initial_node.best_logprob = initial_token.total_logprob

        self._sorted_nodes = lattice.sorted_nodes()
        nodes_processed = 0
        for node in self._sorted_nodes:
            node_tokens = self._tokens[node.id]
            assert node_tokens
            num_pruned_tokens = len(node_tokens)
            self._prune(node)
            node_tokens = self._tokens[node.id]
            assert node_tokens
            num_pruned_tokens -= len(node_tokens)

            if node.id == lattice.final_node.id:
                new_tokens = self._propagate(
                    node_tokens, None, lm_scale, wi_penalty)
                return sorted(new_tokens,
                              key=lambda token: token.total_logprob,
                              reverse=True)

            num_new_tokens = 0
            for link in node.out_links:
                new_tokens = self._propagate(
                    node_tokens, link, lm_scale, wi_penalty)
                self._tokens[link.end_node.id].extend(new_tokens)
                num_new_tokens += len(new_tokens)

            nodes_processed += 1
            if nodes_processed % math.ceil(len(self._sorted_nodes) / 20) == 0:
                logging.debug("[%d] (%.2f %%) -- tokens = %d +%d -%d",
                              nodes_processed,
                              nodes_processed / len(self._sorted_nodes) * 100,
                              len(node_tokens),
                              num_new_tokens,
                              num_pruned_tokens)

        raise InputError("Could not reach the final node of word lattice.")

    def _propagate(self, tokens, link, lm_scale, wi_penalty):
        """Propagates tokens to given link or to end of sentence.

        Lattices may contain !NULL, !ENTER, !EXIT, etc. nodes that model e.g.
        silence or sentence start or end, or for example when the topology is
        easier to represent with extra nodes. Such null nodes may contain
        language model scores. Then the function will update the acoustic and
        lattice LM score, but will not compute anything with the neural network.

        Also updates ``best_logprob`` of the end node, so that beam pruning
        threshold can be obtained efficiently.

        :type tokens: list of LatticeDecoder.Tokens
        :param tokens: input tokens

        :type link: Lattice.Link
        :param link: if other than ``None``, propagates the tokens to this link;
                     if ``None``, just updates the LM logprobs as if the tokens
                     were propagated to an end of sentence

        :type lm_scale: logprob_type
        :param lm_scale: scale language model log probabilities by this factor

        :type wi_penalty: logprob_type
        :param wi_penalty: penalize word insertion by adding this value to the
                           total log probability of the token

        :rtype: list of LatticeDecoder.Tokens
        :returns: the propagated tokens
        """

        new_tokens = [self.Token.copy(token) for token in tokens]

        if link is None:
            self._append_word(new_tokens, self._eos_id)
        else:
            for token in new_tokens:
                token.ac_logprob += link.ac_logprob
                token.lat_lm_logprob += link.lm_logprob
            if not link.word.startswith('!'):
                try:
                    word_id = self._vocabulary.word_to_id[link.word]
                except KeyError:
                    word_id = self._unk_id
                self._append_word(new_tokens, word_id)

        for token in new_tokens:
            token.recompute_hash(self._recombination_order)
            token.recompute_total(self._nnlm_weight, lm_scale, wi_penalty,
                                  self._linear_interpolation)
            if not link is None:
                if (link.end_node.best_logprob is None) or \
                   (token.total_logprob > link.end_node.best_logprob):
                    link.end_node.best_logprob = token.total_logprob

        return new_tokens

    def _prune(self, node):
        """Prunes tokens from a node according to beam and the maximum number of
        tokens.

        :type node: Lattice.Node
        :param node: perform pruning on this node
        """

        new_tokens = dict()
        for token in self._tokens[node.id]:
            key = token.recombination_hash
            if (not key in new_tokens) or \
               (token.total_logprob > new_tokens[key].total_logprob):
                new_tokens[key] = token

        # Sort the tokens by descending log probability.
        new_tokens = sorted(new_tokens.values(),
                            key=lambda token: token.total_logprob, reverse=True)

        # Compare to the best probability at the same or later time.
        if not self._beam is None:
            if node.time is None:
                node_ids = [iter_node.id for iter_node in self._sorted_nodes]
                time_begin = node_ids.index(node.id)
            else:
                for time_begin, iter_node in enumerate(self._sorted_nodes):
                    if (not iter_node.time is None) and \
                       (iter_node.time >= node.time):
                        break
            assert time_begin < len(self._sorted_nodes)

            best_logprob = max(iter_node.best_logprob
                               for iter_node in self._sorted_nodes[time_begin:]
                               if not iter_node.best_logprob is None)
            threshold = best_logprob - self._beam
            token_index = len(new_tokens) - 1
            while (token_index >= 1) and \
                  (new_tokens[token_index].total_logprob <= threshold):
                del new_tokens[token_index]
                token_index -= 1

        # Enforce limit on number of tokens at each node.
        if not self._max_tokens_per_node is None:
            new_tokens[self._max_tokens_per_node:] = []

        self._tokens[node.id] = new_tokens

    def _append_word(self, tokens, target_word_id):
        """Appends a word to each of the given tokens, and updates their scores.

        :type tokens: list of LatticeDecoder.Tokens
        :param tokens: input tokens

        :type target_word_id: int
        :param target_word_id: word ID to be appended to the existing history of
                               each input token
        """

        word_input = [[token.history[-1] for token in tokens]]
        word_input = numpy.asarray(word_input).astype('int64')
        class_input, membership_probs = \
            self._vocabulary.get_class_memberships(word_input)
        recurrent_state = [token.state for token in tokens]
        recurrent_state = RecurrentState.combine_sequences(recurrent_state)
        target_class_ids = numpy.ones(shape=(1, len(tokens))).astype('int64')
        target_class_ids *= self._vocabulary.word_id_to_class_id[target_word_id]
        step_result = self.step_function(word_input,
                                         class_input,
                                         target_class_ids,
                                         *recurrent_state.get())
        logprobs = step_result[0]
        # Add logprobs from the class membership of the predicted words.
        logprobs += numpy.log(membership_probs)
        output_state = step_result[1:]

        for index, token in enumerate(tokens):
            token.history.append(target_word_id)
            token.state = RecurrentState(self._network.recurrent_state_size)
            # Slice the sequence that corresponds to this token.
            token.state.set([layer_state[:,index:index+1]
                             for layer_state in output_state])

            if target_word_id == self._unk_id:
                if self._ignore_unk:
                    continue
                if not self._unk_penalty is None:
                    token.nn_lm_logprob += self._unk_penalty
                    continue
            # logprobs matrix contains only one time step.
            token.nn_lm_logprob += logprobs[0,index]
