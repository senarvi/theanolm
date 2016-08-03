#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
import math
from decimal import *
import logging
import numpy
import theano
from theanolm.network import RecurrentState

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
                     ac_logprob=0.0,
                     lat_lm_logprob=0.0,
                     nn_lm_logprob=0.0):
            """Constructs a token with given recurrent state and logprobs.

            The constructor won't compute the total logprob. The user is
            responsible for computing it when necessary, to avoid unnecessary
            overhead.

            :type history: list of ints
            :param history: word IDs that the token has passed

            :type state: RecurrentState
            :param state: the state of the recurrent layers for a single
                          sequence

            :type ac_logprob: float
            :param ac_logprob: sum of the acoustic log probabilities of the
                               lattice links

            :type lat_lm_logprob: float
            :param lat_lm_logprob: sum of the LM log probabilities of the
                                   lattice links

            :type nn_lm_logprob: float
            :param nn_lm_logprob: sum of the NNLM log probabilities of the
                                  lattice links
            """

            self.history = history
            self.state = state
            self.ac_logprob = ac_logprob
            self.lat_lm_logprob = lat_lm_logprob
            self.nn_lm_logprob = nn_lm_logprob
            self.total_logprob = None

        @classmethod
        def copy(classname, token):
            """Creates a copy of a token.

            The recurrent layer states will not be copied - a pointer will be
            copied instead. There's no need to copy the structure, since we
            never modify the state of a token, but replace it if necessary.

            Total log probability will not be copied.

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

        def interpolate(self, nn_lm_weight, lm_scale, wi_penalty):
            """Computes the interpolated language model log probability and
            the total log probability.

            The interpolated LM log probability is saved in ``self.lm_logprob``.
            The total log probability is computed by applying LM scale factor
            and and adding the acoustic log probability and word insertion
            penalty.

            :type nn_lm_weight: float
            :param nn_lm_weight: weight of the neural network LM probability
                                 when interpolating with the lattice probability

            :type lm_scale: float
            :param lm_scale: scaling factor for LM probability when computing
                             the total probability

            :type wi_penalty: float
            :param wi_penalty: penalize each word in the history by adding this
                               value as many times as there are words
            """

            lat_lm_prob = math.exp(self.lat_lm_logprob)
            nn_lm_prob = math.exp(self.nn_lm_logprob)
            if (lat_lm_prob > 0) and (nn_lm_prob > 0):
                lm_prob = (1.0 - nn_lm_weight) * lat_lm_prob
                lm_prob += nn_lm_weight * nn_lm_prob
                self.lm_logprob = math.log(lm_prob)
            else:
                # An exp() resulted in an underflow. Use the decimal library.
                getcontext().prec = 16
                d_nn_lm_weight = Decimal(nn_lm_weight)
                d_inv_nn_lm_weight = Decimal(1.0) - d_nn_lm_weight
                d_lat_lm_logprob = Decimal(self.lat_lm_logprob)
                d_nn_lm_logprob = Decimal(self.nn_lm_logprob)
                d_lm_prob = d_inv_nn_lm_weight * d_lat_lm_logprob.exp()
                d_lm_prob += d_nn_lm_weight * d_nn_lm_logprob.exp()
                self.lm_logprob = float(d_lm_prob.ln())

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

    def __init__(self, network, nnlm_weight=1.0, lm_scale=None, wi_penalty=None,
                 ignore_unk=False, unk_penalty=None, max_tokens_per_node=None,
                 profile=False):
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

        :type network: Network
        :param network: the neural network object

        :type nnlm_weight: float
        :param nnlm_weight: weight of the neural network probabilities when
                            interpolating with the lattice probabilities

        :type lm_scale: float
        :param lm_scale: if other than ``None``, the decoder will scale language
                         model log probabilities by this factor; otherwise the
                         scaling factor will be read from the lattice file

        :type wi_penalty: float
        :param wi_penalty: penalize word insertion by adding this value to the
                           total log probability of a token as many times as
                           there are words

        :type ignore_unk: bool
        :param ignore_unk: if set to True, <unk> tokens are excluded from
                           perplexity computation

        :type unk_penalty: float
        :param unk_penalty: if set to othern than None, used as <unk> token
                            score

        :type max_tokens_per_node: int
        :param max_tokens_per_node: if set to othern than None, leave only this
                                    many tokens at each node

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        self._network = network
        self._vocabulary = network.vocabulary
        self._nnlm_weight = nnlm_weight
        self._lm_scale = lm_scale
        self._wi_penalty = wi_penalty
        self._max_tokens_per_node = max_tokens_per_node

        self._sos_id = self._vocabulary.word_to_id['<s>']
        self._eos_id = self._vocabulary.word_to_id['</s>']

        inputs = [network.word_input,
                  network.class_input,
                  network.target_class_ids]
        inputs.extend(network.recurrent_state_input)

        outputs = [network.target_probs()]
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
            lm_scale = self._lm_scale
        elif not lattice.lm_scale is None:
            lm_scale = lattice.lm_scale
        else:
            lm_scale = 1.0

        if not self._wi_penalty is None:
            wi_penalty = self._wi_penalty
        if not lattice.wi_penalty is None:
            wi_penalty = lattice.wi_penalty
        else:
            wi_penalty = 0.0

        tokens = [list() for _ in lattice.nodes]
        initial_state = RecurrentState(self._network.recurrent_state_size)
        initial_token = self.Token(history=[self._sos_id], state=initial_state)
        initial_token.interpolate(self._nnlm_weight, lm_scale, wi_penalty)
        tokens[lattice.initial_node.id].append(initial_token)

        sorted_nodes = lattice.sorted_nodes()
        nodes_processed = 0
        for node in sorted_nodes:
            node_tokens = tokens[node.id]
            assert node_tokens

            if node.id == lattice.final_node.id:
                new_tokens = self._propagate(
                    node_tokens, None, lm_scale, wi_penalty)
                return sorted(new_tokens,
                              key=lambda token: token.total_logprob,
                              reverse=True)
            for link in node.out_links:
                new_tokens = self._propagate(
                    node_tokens, link, lm_scale, wi_penalty)
                tokens[link.end_node.id].extend(new_tokens)
                if not self._max_tokens_per_node is None:
                    # Enforce limit on number of tokens at each node.
                    tokens[link.end_node.id].sort(
                        key=lambda token: token.total_logprob,
                        reverse=True)
                    tokens[link.end_node.id][self._max_tokens_per_node:] = []

            nodes_processed += 1
            if nodes_processed % math.ceil(len(sorted_nodes) / 20) == 0:
                logging.debug("[%d] (%.2f %%) -- tokens = %d",
                              nodes_processed,
                              nodes_processed / len(sorted_nodes) * 100,
                              len(node_tokens))

        raise InputError("Could not reach the final node of word lattice.")

    def _propagate(self, tokens, link, lm_scale, wi_penalty):
        """Propagates tokens to given link or to end of sentence.

        Lattices may contain !NULL, !ENTER, !EXIT, etc. nodes that model e.g.
        silence or sentence start or end, or for example when the topology is
        easier to represent with extra nodes. Such null nodes may contain
        language model scores. Then the function will update the acoustic and
        lattice LM score, but will not compute anything with the neural network.

        :type tokens: list of LatticeDecoder.Tokens
        :param tokens: input tokens

        :type link: Lattice.Link
        :param link: if other than ``None``, propagates the tokens to this link;
                     if ``None``, just updates the LM logprobs as if the tokens
                     were propagated to an end of sentence

        :type lm_scale: float
        :param lm_scale: scale language model log probabilities by this factor

        :type wi_penalty: float
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
                word_id = self._vocabulary.word_to_id[link.word]
                self._append_word(new_tokens, word_id)

        for token in new_tokens:
            token.interpolate(self._nnlm_weight, lm_scale, wi_penalty)

        return new_tokens

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
        class_input = self._vocabulary.word_id_to_class_id[word_input]
        recurrent_state = [token.state for token in tokens]
        recurrent_state = RecurrentState.combine_sequences(recurrent_state)
        target_class_ids = numpy.ones(shape=(1, len(tokens))).astype('int64')
        target_class_ids *= self._vocabulary.word_id_to_class_id[target_word_id]
        step_result = self.step_function(word_input,
                                         class_input,
                                         target_class_ids,
                                         *recurrent_state.get())
        output_logprobs = step_result[0]
        output_state = step_result[1:]

        for index, token in enumerate(tokens):
            token.history.append(target_word_id)
            token.state = RecurrentState(self._network.recurrent_state_size)
            # Slice the sequence that corresponds to this token.
            token.state.set([layer_state[:,index:index+1]
                             for layer_state in output_state])
            # The matrix contains only one time step.
            token.nn_lm_logprob += output_logprobs[0,index]
