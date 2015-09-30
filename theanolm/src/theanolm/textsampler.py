#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class TextSampler(object):
    """Neural network language model sampler

    A Theano function that generates text using a neural network language
    model.
    """

    def __init__(self, network, dictionary):
        """Creates the neural network architecture.

        Creates the function self.step_function that uses the state of the
        previous time step and the word ID of the current time step, to compute
        the output distribution. It samples from the output distribution and
        returns the sampled word ID along with the output state of this time
        step.

        :type network: Network
        :param network: the neural network object
        """

        self.network = network
        self.dictionary = dictionary
        self.trng = RandomStreams(1234)

        inputs = [self.network.onestep_input]
        inputs.extend(self.network.onestep_state)

        word_probs = self.network.onestep_output
        word_ids = self.trng.multinomial(pvals=word_probs).argmax(1)
        outputs = [word_ids]
        outputs.extend(self.network.hidden_layer.onestep_outputs)

        self.step_function = theano.function(
            inputs, outputs, name='text_sampler')

    def generate(self, max_length=30):
        """ Generates a text sequence.

        Calls self.step_function() repeatedly, reading the word output and
        the state output of the hidden layer and passing the hidden layer state
        output to the next time step.

        :rtype: list of strs
        :returns: list of the generated words
        """

        # -1 indicates the first word of a sequence. We are only generating one
        # sequence at a time.
        word_ids = -1 * numpy.ones(shape=(1,)).astype('int64')

        # Construct a list of hidden layer state variables and initialize them
        # to zeros. GRU has only one state that travels through the time steps,
        # LSTM has two.
        hidden_state_shape = (1, self.network.architecture.hidden_layer_size)
        hidden_layer_state = [
            numpy.zeros(shape=hidden_state_shape).astype('float32')
            for _ in range(self.network.hidden_layer.num_state_variables)]

        result = []
        for _ in range(max_length):
            step_result = self.step_function(word_ids, *hidden_layer_state)
            word_ids = step_result[0]
            hidden_layer_state = step_result[1:]
            word_id = word_ids[0]
            result.append(word_id)
            if word_id == 0:
                break
        return self.dictionary.ids_to_words(result)
