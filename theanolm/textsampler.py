#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import theano
from theanolm.network import RecurrentState

class TextSampler(object):
    """Neural network language model sampler

    A Theano function that generates text using a neural network language
    model.
    """

    def __init__(self, network):
        """Creates a Theano function that samples one word at a time.

        Creates the function self.step_function that uses the state of the
        previous time step and the word ID of the current time step, to compute
        the output distribution. It samples from the output distribution and
        returns the sampled word ID along with the output state of this time
        step.

        :type network: Network
        :param network: the neural network object
        """

        self.network = network
        self.vocabulary = network.vocabulary
        self.random = self.network.random

        inputs = [self.network.word_input, self.network.class_input]
        inputs.extend(self.network.recurrent_state_input)

        # multinomial() is only implemented with dimension < 2, but the matrix
        # contains only one time step anyway.
        output_probs = self.network.output_probs()[0]
        class_ids = self.random.multinomial(pvals=output_probs).argmax(1)
        class_ids = class_ids.reshape([1, class_ids.shape[0]])
        outputs = [class_ids]
        outputs.extend(self.network.recurrent_state_output)

        # Ignore unused input, because is_training is only used by dropout
        # layer.
        self.step_function = theano.function(
            inputs,
            outputs,
            givens=[(self.network.is_training, numpy.int8(0))],
            name='text_sampler',
            on_unused_input='ignore')

    def generate(self, max_length=30, num_sequences=1):
        """Generates a text sequence.

        Calls self.step_function() repeatedly, reading the word output and
        the state output of the hidden layer and passing the hidden layer state
        output to the next time step.

        Generates at most ``max_length`` words, stopping if a sentence break is
        generated.

        :type max_length: int
        :param max_length: maximum number of words in a sequence

        :type num_sequences: int
        :param num_sequences: number of sequences to generate in parallel

        :rtype: list of list of strs
        :returns: list of word sequences
        """

        sos_id = self.vocabulary.word_to_id['<s>']
        sos_class_id = self.vocabulary.word_id_to_class_id[sos_id]
        eos_id = self.vocabulary.word_to_id['</s>']

        word_input = sos_id * \
                     numpy.ones(shape=(1, num_sequences)).astype('int64')
        class_input = sos_class_id * \
                      numpy.ones(shape=(1, num_sequences)).astype('int64')
        result = sos_id * \
                 numpy.ones(shape=(max_length, num_sequences)).astype('int64')
        state = RecurrentState(self.network, num_sequences)

        for time_step in range(1, max_length):
            # The input is the output from the previous step.
            step_result = self.step_function(word_input,
                                             class_input,
                                             *state.get())
            class_ids = step_result[0]
            # The class IDs from the single time step.
            step_class_ids = class_ids[0]
            step_word_ids = numpy.array(
                self.vocabulary.class_ids_to_word_ids(step_class_ids))
            result[time_step] = step_word_ids
            word_input = step_word_ids[numpy.newaxis]
            class_input = class_ids
            state.set(step_result[1:])

        return self.vocabulary.id_to_word[result.transpose()].tolist()
