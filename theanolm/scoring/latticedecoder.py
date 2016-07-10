#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
import math
from decimal import *
import logging
from theanolm.network import RecurrentState

class LatticeDecoder(object):
    """Word Lattice Decoding Using a Neural Network Language Model
    """

    class Token:
        def __init__(self,
                     history=[],
                     state=[],
                     ac_logprob=0.0,
                     lat_lm_logprob=0.0,
                     nn_lm_logprob=0.0):
            """Constructs a token with given recurrent state and logprobs.

            :type ac_logprob: float
            :param ac_logprob: sum of the acoustic log probabilities of the
                               lattice links

            :type lat_lm_logprob: float
            :param lat_lm_logprob: sum of the LM log probabilities of the
                                   lattice links
            """

            self.history = history
            self.state = state
            self.ac_logprob = ac_logprob
            self.lat_lm_logprob = lat_lm_logprob
            self.nn_lm_logprob = nn_lm_logprob

        def compute_total_logprob(self, nn_lm_weight=0.5, lm_scale=1.0):
            """Computes the total log probability of the token by interpolating
            the LM logprobs, applying LM scale, and adding the acoustic logprob.
            """

            lat_lm_prob = math.exp(self.lat_lm_logprob)
            nn_lm_prob = math.exp(self.nn_lm_logprob)
            if (lat_lm_prob > 0) and (nn_lm_prob > 0):
                lm_prob = (1.0 - nn_lm_weight) * lat_lm_prob
                lm_prob += nn_lm_weight * nn_lm_prob
                lm_logprob = math.log(lm_prob)
            else:
                # An exp() resulted in an underflow. Use the decimal library.
                getcontext().prec = 16
                d_nn_lm_weight = Decimal(nn_lm_weight)
                d_inv_nn_lm_weight = Decimal(1.0) - d_nn_lm_weight
                d_lat_lm_logprob = Decimal(self.lat_lm_logprob)
                d_nn_lm_logprob = Decimal(self.nn_lm_logprob)
                d_lm_prob = d_inv_nn_lm_weight * d_lat_lm_logprob.exp()
                d_lm_prob += d_nn_lm_weight * d_nn_lm_logprob.exp()
                lm_logprob = float(d_lm_prob.ln())
            self.total_logprob = self.ac_logprob + (lm_logprob * lm_scale)

    def __init__(self, network, weight):
        """Constructs a decoder.

        :type weight: float
        :param weight: weight of the neural network probabilities when
                       interpolating with the lattice probabilities
        """

        self._network = network
        self._vocabulary = network.vocabulary
        self._weight = weight

    def decode(self, lattice):
        """Decodes the best sentence from a lattice.
        """

        self.tokens = [list() for _ in lattice.nodes]
        initial_state = RecurrentState(self._network)
        initial_token = Token(history=['<s>'], state=initial_state)
        self.tokens[lattice.initial_node.id].append(initial_token)

        sorted_nodes = lattice.sorted_nodes()
        for node in sorted_nodes:
            node_tokens = self.tokens[node.id]
            assert node_tokens
            while node_tokens:
                token = node_tokens.pop()
                if node.out_links:
                    for link in node.out_links:
                        new_token = self._pass_token(token, link)
                        self.tokens[link.end_node.id].append(new_token)
                else:
                    self._pass_token_to_eos(token)

    def _pass_token(self, token, link):
        """Passes a copy of ``token`` to a link.

        Lattices may contain !NULL, !ENTER, !EXIT, etc. nodes at sentence starts
        and ends, and for example when the topology is easier to represent with
        extra nodes. Such null nodes may contain language model scores. Then the
        function will update the lattice LM score, but will not do anything with
        the neural network.
        """

        new_token = self.Token(history=deepcopy(token.history),
                               ac_logprob=token.ac_logprob,
                               lat_lm_logprob=token.lat_lm_logprob,
                               nn_lm_logprob=token.nn_lm_logprob)
        new_token.ac_logprob += link.ac_logprob
        new_token.lat_lm_logprob += link.lm_logprob

        if link.word.startswith('!'):
            # I don't think we need to copy the state since we're not modifying
            # it.
            new_token.state = token.state
        else:
            input_word_id = token.history[-1]
            input_class_id = self._vocabulary.word_id_to_class_id[word_id]

            target_word_id = self._vocabulary.word_to_id[link.word]
            new_token.history.append(target_word_id)

            singleton = numpy.ones(shape=(1, 1)).astype('int64')
            word_input = input_word_id * singleton
            class_input = input_class_id * singleton
            step_result = self.step_function(word_input,
                                             class_input,
                                             *token.state)
            output_logprobs = step_result[0]
            output_state = step_result[1:]
            new_token.nn_lm_logprob += target_prob(output_logprobs, target_word_id) # XXX
            new_token.state = output_state

        new_token.compute_total_logprob()
        return new_token

    def _pass_token_to_eos(self, token):
        """Updates token with a pass to end of sentence.
        """

        input_word_id = token.history[-1]
        input_class_id = self._vocabulary.word_id_to_class_id[word_id]

        target_word_id = self._vocabulary.word_to_id['</s>']
        token.history.append(target_word_id)

        singleton = numpy.ones(shape=(1, 1)).astype('int64')
        word_input = input_word_id * singleton
        class_input = input_class_id * singleton
        step_result = self.step_function(word_input,
                                         class_input,
                                         *token.state)
        output_logprobs = step_result[0]
        output_state = step_result[1:]
        token.nn_lm_logprob += target_prob(output_logprobs, target_word_id) # XXX
        token.state = output_state
