#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the LatticeDecoder class.
"""

import logging
import math

import numpy
import theano
from theano import tensor

from theanolm.backend import InputError
from theanolm.backend import interpolate_linear, interpolate_loglinear
from theanolm.backend import logprob_type
from theanolm.network import RecurrentState

class LatticeDecoder(object):
    """Word Lattice Decoding Using a Neural Network Language Model
    """

    class Token:
        """Decoding Token

        A token represents a partial path through a word lattice. The decoder
        propagates a set of tokens through the lattice by
        """
        __slots__ = ("history", "state", "ac_logprob", "lat_lm_logprob",
                     "nn_lm_logprob", "recombination_hash", "graph_logprob",
                     "total_logprob")

        def __init__(self,
                     history=(),
                     state=None,
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
            self.state = [] if state is None else state
            self.ac_logprob = ac_logprob
            self.lat_lm_logprob = lat_lm_logprob
            self.nn_lm_logprob = nn_lm_logprob
            self.recombination_hash = None
            self.graph_logprob = None
            self.total_logprob = None

        @classmethod
        def copy(cls, token):
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

            return cls(token.history,
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
            self.recombination_hash = hash(limited_history)

        def recompute_total(self, nn_lm_weight, lm_scale, wi_penalty,
                            linear=False):
            """Computes the interpolated language model log probability and
            the total log probability.

            First the interpolated LM log probability is computed. Then the
            total log probability is computed by applying the LM scale factor,
            and adding the acoustic log probability and the word insertion
            penalty.

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
                lm_logprob = interpolate_linear(
                    self.nn_lm_logprob, self.lat_lm_logprob,
                    nn_lm_weight)
            else:
                lm_logprob = interpolate_loglinear(
                    self.nn_lm_logprob, self.lat_lm_logprob,
                    nn_lm_weight, (1.0 - nn_lm_weight))

            self.graph_logprob = lm_logprob * lm_scale
            self.graph_logprob += wi_penalty * (len(self.history) - 1)

            self.total_logprob = self.ac_logprob + self.graph_logprob

        def history_words(self, vocabulary):
            """Converts the word IDs in the history to words using
            ``vocabulary``. The history may contain also OOV words as text, so
            any ``str`` will be left untouched.

            :type vocabulary: Vocabulary
            :param vocabulary: mapping from word IDs to words

            :rtype: list of strs
            :returns: the token's history as list of words
            """

            return [vocabulary.id_to_word[word] if isinstance(word, int)
                    else word
                    for word in self.history]

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
                history = ' '.join(self.history_words(vocabulary))

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

        Creates the function self._step_function that takes as input a set of
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

        unk_penalty : float
          if set to other than None, used as <unk> token score

        use_shortlist : bool
          if set to ``True``, <unk> token probability is distributed among the
          out-of-shortlist words according to their unigram probabilities

        unk_from_lattice : bool
          if set to ``True``, the probability for <unk> tokens is taken from the
          lattice alone

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

        prune_extra_limit : float
          if set, adjust the beam and max_tokens_per_node pruning relative to
          the number of tokens; the limits are divided by the number of tokens
          and multiplied by this factor

        abs_min_beam : float
          when using prune_extra_limit, this is the minimum that the beam will
          be adjusted to

        abs_min_max_tokens : float
          when using prune_extra_limit, this is the minimum that the maximum
          number of tokens will be adjusted to

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
        self._unk_penalty = decoding_options['unk_penalty']
        self._unk_from_lattice = decoding_options['unk_from_lattice']
        self._linear_interpolation = decoding_options['linear_interpolation']
        self._max_tokens_per_node = decoding_options['max_tokens_per_node']
        self._beam = decoding_options['beam']
        if self._beam is not None:
            self._beam = logprob_type(self._beam)
        self._recombination_order = decoding_options['recombination_order']
        self._prune_extra_limit = decoding_options.get('prune_extra_limit', None)
        self._abs_min_beam = decoding_options.get('abs_min_beam', 0)
        self._abs_min_max_tokens = decoding_options.get('abs_min_max_tokens', 0)
        if self._prune_extra_limit is None:
            self._abs_min_beam = self._abs_min_max_tokens = 0

        if decoding_options['use_shortlist'] and \
           self._vocabulary.has_unigram_probs():
            oos_logprobs = numpy.log(self._vocabulary.get_oos_probs())
            self._oos_logprobs = oos_logprobs.astype(theano.config.floatX)
        else:
            self._oos_logprobs = None

        self._sos_id = self._vocabulary.word_to_id['<s>']
        self._eos_id = self._vocabulary.word_to_id['</s>']
        self._unk_id = self._vocabulary.word_to_id['<unk>']

        inputs = [network.input_word_ids,
                  network.input_class_ids,
                  network.target_class_ids]
        inputs.extend(network.recurrent_state_input)

        outputs = [tensor.log(network.target_probs())]
        outputs.extend(network.recurrent_state_output)

        # Ignore unused input, because is_training is only used by dropout
        # layer.
        self._step_function = theano.function(
            inputs,
            outputs,
            givens=[(network.is_training, numpy.int8(0))],
            name='step_predictor',
            profile=profile,
            on_unused_input='ignore')

    def decode(self, lattice):
        """Propagates tokens through given lattice and returns a list of tokens
        in the final nodes.

        Propagates tokens at a node to every outgoing link by creating a copy of
        each token and updating the language model scores according to the link.

        The function returns two lists. The first list contains the final
        tokens, sorted in the descending order of total log probability. I.e.
        the first token in the list represents the best path through the
        lattice. The second list contains the tokens that were dropped during
        recombination. This is needed for constructing a new rescored lattice.

        :type lattice: Lattice
        :param lattice: a word lattice to be decoded

        :rtype: a tuple of two lists of LatticeDecoder.Tokens
        :returns: a list of the final tokens sorted by probability (most likely
                  token first), and a list of the tokens that were dropped
                  during recombination
        """

        if self._lm_scale is not None:
            lm_scale = logprob_type(self._lm_scale)
        elif lattice.lm_scale is not None:
            lm_scale = logprob_type(lattice.lm_scale)
        else:
            lm_scale = logprob_type(1.0)

        if self._wi_penalty is not None:
            wi_penalty = logprob_type(self._wi_penalty)
        elif lattice.wi_penalty is not None:
            wi_penalty = logprob_type(lattice.wi_penalty)
        else:
            wi_penalty = logprob_type(0.0)

        tokens = [list() for _ in lattice.nodes]
        recomb_tokens = []
        initial_state = RecurrentState(self._network.recurrent_state_size)
        initial_token = self.Token(history=(self._sos_id,), state=initial_state)
        initial_token.recompute_hash(self._recombination_order)
        initial_token.recompute_total(self._nnlm_weight, lm_scale, wi_penalty,
                                      self._linear_interpolation)
        tokens[lattice.initial_node.id].append(initial_token)
        lattice.initial_node.best_logprob = initial_token.total_logprob

        sorted_nodes = lattice.sorted_nodes()
        self._nodes_processed = 0
        final_tokens = []
        for node in sorted_nodes:
            stats = self._prune(node, sorted_nodes, tokens, recomb_tokens)

            num_new_tokens = 0
            node_tokens = tokens[node.id]
            assert node_tokens
            if node.final:
                new_tokens = self._propagate(
                    node_tokens, None, lm_scale, wi_penalty)
                final_tokens.extend(new_tokens)
                num_new_tokens += len(new_tokens)
            for link in node.out_links:
                new_tokens = self._propagate(
                    node_tokens, link, lm_scale, wi_penalty)
                tokens[link.end_node.id].extend(new_tokens)
                # If there are lots of tokens in the end node, prune already to
                # conserve memory.
                if self._max_tokens_per_node is not None and \
                   len(tokens[link.end_node.id]) > self._max_tokens_per_node * 2:
                    self._prune(link.end_node, sorted_nodes, tokens, recomb_tokens)
                num_new_tokens += len(new_tokens)
            stats['new'] = num_new_tokens

            self._nodes_processed += 1
            self._log_stats(stats, node.id, len(sorted_nodes))

        if len(final_tokens) == 0:
            raise InputError("Could not reach a final node of word lattice.")

        final_tokens = self._sorted_recombined_tokens(final_tokens,
                                                      recomb_tokens)
        return final_tokens, recomb_tokens

    def _propagate(self, tokens, link, lm_scale, wi_penalty):
        """Propagates tokens to given link or to end of sentence.

        Lattices may contain null nodes with word ``None`` that model e.g.
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
                if link.ac_logprob is not None:
                    token.ac_logprob += link.ac_logprob
                if link.lm_logprob is not None:
                    token.lat_lm_logprob += link.lm_logprob

            if link.word is not None:
                try:
                    word = self._vocabulary.word_to_id[link.word]
                except KeyError:
                    word = link.word
                if self._unk_from_lattice:
                    self._append_word(new_tokens, word, link.lm_logprob)
                elif self._unk_penalty is not None:
                    self._append_word(new_tokens, word, self._unk_penalty)
                else:
                    self._append_word(new_tokens, word)

        for token in new_tokens:
            token.recompute_hash(self._recombination_order)
            token.recompute_total(self._nnlm_weight, lm_scale, wi_penalty,
                                  self._linear_interpolation)
            if link is not None and link.end_node is not None:
                if (link.end_node.best_logprob is None) or \
                   (token.total_logprob > link.end_node.best_logprob):
                    link.end_node.best_logprob = token.total_logprob

        return new_tokens

    def _prune(self, node, sorted_nodes, tokens, recomb_tokens):
        """Prunes tokens from a node according to beam and the maximum number of
        tokens, and recombines tokens whose N previous words are identical
        (where N is the recombination order).

        :type node: Lattice.Node
        :param node: perform pruning on this node

        :type sorted_nodes: list of Lattice.Nodes
        :param sorted_nodes: all nodes in topological order

        :type tokens: list of LatticeDecoder.Tokens
        :param tokens: all tokens

        :type recomb_tokens: list of tuples
        :param recomb_tokens: for all tokens that were dropped during
            recombination, contains the token, and the history and NNLM log
            probability of the kept token; dropped tokens will be added to the
            list

        :rtype: dict
        :returns: a dictionary of statistics collected during pruning
        """

        node_tokens = tokens[node.id]
        assert node_tokens
        stats = dict()
        stats['before'] = len(node_tokens)

        limit_divider = 1
        if self._prune_extra_limit is not None:
            limit_divider = len(sorted_nodes) // self._prune_extra_limit + 1
            limit_divider = max(1, limit_divider)

        new_tokens = self._sorted_recombined_tokens(node_tokens,
                                                    recomb_tokens)
        stats['after-recomb'] = len(new_tokens)

        # Compare to the best probability at the same or later time.
        if self._beam is not None:
            if node.time is None:
                node_ids = [iter_node.id for iter_node in sorted_nodes]
                time_begin = node_ids.index(node.id)
            else:
                for time_begin, iter_node in enumerate(sorted_nodes):
                    if (iter_node.time is not None) and \
                       (iter_node.time >= node.time):
                        break
            assert time_begin < len(sorted_nodes)
            best_logprob = max(iter_node.best_logprob
                               for iter_node in sorted_nodes[time_begin:]
                               if iter_node.best_logprob is not None)

            beam = self._beam / limit_divider
            beam = max(beam, self._abs_min_beam)
            threshold = best_logprob - beam

            stats['best'] = best_logprob
            stats['node-best'] = new_tokens[0].total_logprob
            stats['node-worst'] = new_tokens[-1].total_logprob
            stats['threshold'] = threshold
            stats['average'] = sum(t.total_logprob for t in new_tokens) / len(new_tokens)
            if len(new_tokens) >= 10:
                stats['pos10'] = new_tokens[9].total_logprob
            if len(new_tokens) >= 50:
                stats['pos50'] = new_tokens[49].total_logprob
            if len(new_tokens) >= 100:
                stats['pos100'] = new_tokens[99].total_logprob

            keep_tokens = len(new_tokens)
            while (keep_tokens > 1) and \
                  (new_tokens[keep_tokens - 1].total_logprob <= threshold):
                keep_tokens -= 1
            new_tokens = new_tokens[:keep_tokens]
        stats['after-beam'] = len(new_tokens)

        # Enforce limit on number of tokens at each node.
        if self._max_tokens_per_node is not None:
            max_tokens = self._max_tokens_per_node // limit_divider
            max_tokens = max(max_tokens, self._abs_min_max_tokens)
            new_tokens = new_tokens[:max_tokens]
        stats['after-max'] = len(new_tokens)

        tokens[node.id] = new_tokens
        return stats

    def _sorted_recombined_tokens(self, tokens, recomb_tokens):
        """Sorts tokens by descending probability and recombines tokens with
        identical hash.

        First sorts the tokens by probability, most likely token first. Then
        keeps only the first if there are multiple tokens with the same hash.
        Tokens with identical N previous words in the history (where N is the
        recombination order) will be dropped and added to recomb_tokens.

        :type tokens: list of LatticeDecoder.Tokens
        :param tokens: input tokens

        :type recomb_tokens: list of tuples
        :param recomb_tokens: for all tokens that were dropped during
            recombination, contains the token, and the history and NNLM log
            probability of the kept token; dropped tokens will be added to the
            list

        :rtype: list of LatticeDecoder.Tokens
        :returns: the tokens in sorted order and with only the best token of
                  those with identical recombination hash
        """

        sorted_tokens = sorted(tokens,
                               key=lambda token: token.total_logprob,
                               reverse=True)

        token_map = dict()
        result = []
        for token in sorted_tokens:
            key = token.recombination_hash
            if key not in token_map:
                result.append(token)
                token_map[key] = token
            else:
                kept_token = token_map[key]
                recomb_tokens.append((token,
                                      kept_token.history,
                                      kept_token.nn_lm_logprob))
        return result

    def _append_word(self, tokens, target_word, oov_logprob=None):
        """Appends a word to each of the given tokens, and updates their scores.

        :type tokens: list of LatticeDecoder.Tokens
        :param tokens: input tokens

        :type target_word: int or str
        :param target_word: word ID or word to be appended to the existing
                            history of each input token; if not an integer, the
                            word will be considered ``<unk>`` and this variable
                            will be taken literally as the word that will be
                            used in the resulting transcript

        :type oov_logprob: float
        :param oov_logprob: log probability to be assigned to OOV words
        """

        def limit_to_shortlist(self, word):
            """Returns the ``<unk>`` word ID if the argument is not a shortlist
            word ID.
            """
            if isinstance(word, int) and self._vocabulary.in_shortlist(word):
                return word
            else:
                return self._unk_id

        input_word_ids = [[limit_to_shortlist(self, token.history[-1])
                           for token in tokens]]
        input_word_ids = numpy.asarray(input_word_ids).astype('int64')
        input_class_ids, membership_probs = \
            self._vocabulary.get_class_memberships(input_word_ids)
        recurrent_state = [token.state for token in tokens]
        recurrent_state = RecurrentState.combine_sequences(recurrent_state)
        target_word_id = limit_to_shortlist(self, target_word)
        target_class_ids = numpy.ones(shape=(1, len(tokens))).astype('int64')
        target_class_ids *= self._vocabulary.word_id_to_class_id[target_word_id]
        step_result = self._step_function(input_word_ids,
                                          input_class_ids,
                                          target_class_ids,
                                          *recurrent_state.get())
        logprobs = step_result[0]
        # Add logprobs from the class membership of the predicted words.
        logprobs += numpy.log(membership_probs)
        output_state = step_result[1:]

        for index, token in enumerate(tokens):
            token.history = token.history + (target_word,)
            token.state = RecurrentState(self._network.recurrent_state_size)
            # Slice the sequence that corresponds to this token.
            token.state.set([layer_state[:, index:index + 1]
                             for layer_state in output_state])
            # logprobs matrix contains only one time step.
            token.nn_lm_logprob += self._handle_unk_logprob(target_word,
                                                            logprobs[0, index],
                                                            oov_logprob)

    def _handle_unk_logprob(self, word, network_logprob, oov_logprob):
        """Returns the log probability after applying <unk> processing.

        If ``self._oos_logprobs`` is set and the word is in vocabulary, the
        corresponding value will be added to the network log probability. In
        effect, the probability of out-of-shortlist words (which is the <unk>
        probability) is multiplied by the fraction of the actual word within the
        set of OOS words. For out-of-vocabulary words returns ``oov_logprob`` or
        the value predicted by the network.

        Otherwise, for both out-of-shortlist and out-of-vocabulary words returns
        ``oov_logprob`` or the value predicted by the network.

        For shortlist words returns the value predicted by the network.

        :type word: int or str
        :param word: target word ID or word; if not an integer, the word will be
                     considered ``<unk>``

        :type network_logprob: float
        :param network_logprob: log probability predicted by the network

        :type oov_logprob: float
        :param oov_logprob: log probability to be assigned to OOV words
        """

        in_vocabulary = isinstance(word, int)
        in_shortlist = in_vocabulary and self._vocabulary.in_shortlist(word)

        if self._oos_logprobs is not None:
            if in_vocabulary:
                return network_logprob + self._oos_logprobs[word]
            elif oov_logprob is not None:
                logging.debug("Replacing <unk> logprob %f with %f.",
                              network_logprob, oov_logprob)
                return oov_logprob
        elif (not in_shortlist) and (oov_logprob is not None):
            logging.debug("Replacing <unk> logprob %f with %f.",
                          network_logprob, oov_logprob)
            return oov_logprob

        return network_logprob

    def _log_stats(self, stats, node_id, num_nodes):
        """Writes pruning statistics to debug log.

        :type stats: dict
        :param stats: a dictionary of statistics on the number of tokens and
                      their log probabilities
        """

        if self._nodes_processed % math.ceil(num_nodes / 20) != 0:
            return

        optional = ''
        for name in ('after-recomb', 'after-beam', 'after-max', 'new'):
            if name in stats:
                optional += ' {}={}'.format(name, stats[name])
        logging.debug('[%d] (%.2f %%) node=%d -- tokens before=%d%s',
                      self._nodes_processed,
                      self._nodes_processed / num_nodes * 100,
                      node_id,
                      stats['before'],
                      optional)

        if 'best' in stats:
            optional = ''
            for name in ('node-best', 'node-worst', 'threshold', 'average',
                         'pos10', 'pos50', 'pos100'):
                if name in stats:
                    optional += ' {}={:.1f}'.format(name, stats[name])
            logging.debug('logprob best=%.1f%s',
                          stats['best'],
                          optional)
