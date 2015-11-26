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

        :type dictionary: Dictionary
        :param dictionary: dictionary that provides mapping between words and
                           word IDs
        """

        self.network = network
        self.dictionary = dictionary

        M1 = 2147483647
        M2 = 2147462579
        random_seed = [
            numpy.random.randint(0, M1),
            numpy.random.randint(0, M1),
            numpy.random.randint(1, M1),
            numpy.random.randint(0, M2),
            numpy.random.randint(0, M2),
            numpy.random.randint(1, M2)]
        random = RandomStreams(random_seed)

        inputs = [self.network.onestep_input]
        inputs.extend(self.network.onestep_state)

        word_probs = self.network.onestep_output
        word_ids = random.multinomial(pvals=word_probs).argmax(1)
        outputs = [word_ids]
        outputs.extend(self.network.hidden_layer.onestep_outputs)

        self.step_function = theano.function(
            inputs, outputs, name='text_sampler')

    def generate(self, max_length=30):
        """ Generates a text sequence.

        Calls self.step_function() repeatedly, reading the word output and
        the state output of the hidden layer and passing the hidden layer state
        output to the next time step.

        Generates at most ``max_length`` words, stopping if a sentence break is
        generated.

        :type max_length: int
        :param max_length: maximum number of words to generate

        :rtype: list of strs
        :returns: list of the generated words
        """

        # We are only generating one sequence at a time.
        result = [self.dictionary.sos_id]
        previous_step_output = \
            self.dictionary.sos_id * numpy.ones(shape=(1,)).astype('int64')

        # Construct a list of hidden layer state variables and initialize them
        # to zeros. GRU has only one state that is passed through the time
        # steps, LSTM has two.
        hidden_state_shape = (1, self.network.architecture.hidden_layer_size)
        hidden_layer_state = [
            numpy.zeros(shape=hidden_state_shape).astype(theano.config.floatX)
            for _ in range(self.network.hidden_layer.num_state_variables)]

        for _ in range(max_length):
            step_result = self.step_function(previous_step_output,
                                             *hidden_layer_state)
            previous_step_output = step_result[0]
            hidden_layer_state = step_result[1:]
            word_id = previous_step_output[0]
            result.append(word_id)
            if word_id == self.dictionary.eos_id:
                break
        return self.dictionary.ids_to_words(result)
