#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import theano

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

        inputs = [self.network.input]
        inputs.extend(self.network.recurrent_state_input)

        # multinomial() is only implemented with dimension < 2, but the matrix
        # contains only one time step anyway.
        word_probs = self.network.output[0]
        word_ids = self.network.random.multinomial(pvals=word_probs).argmax(1)
        word_ids = word_ids.reshape([1, word_ids.shape[0]])
        outputs = [word_ids]
        outputs.extend(self.network.recurrent_state_output)

        # Ignore unused input, because is_training is only used by dropout
        # layer.
        self.step_function = theano.function(
            inputs,
            outputs,
            givens=[(self.network.is_training, numpy.int8(0))],
            name='text_sampler',
            on_unused_input='ignore')

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

        # We are only generating one sequence at a time. The input is passed as
        # a 2-dimensional matrix with only one element, since in mini-batch mode
        # the matrix contains multiple sequences and time steps.
        result = [self.dictionary.sos_id]
        step_output = \
            self.dictionary.sos_id * numpy.ones(shape=(1,1)).astype('int64')

        # Construct a list of recurrent state variables that will be passed
        # through time steps, and initialize them to zeros. The state vector is
        # specific to sequence and time step, but in this case we have only one
        # sequence and time step.
        recurrent_state = []
        for size in self.network.recurrent_state_size:
            shape = (1, 1, size)
            value = numpy.zeros(shape).astype(theano.config.floatX)
            recurrent_state.append(value)

        for _ in range(max_length):
            # The input is the output from the previous step.
            step_result = self.step_function(step_output,
                                             *recurrent_state)
            step_output = step_result[0]
            recurrent_state = step_result[1:]
            # The word ID from the single time step from the single sequence.
            word_id = step_output[0,0]
            result.append(word_id)
            if word_id == self.dictionary.eos_id:
                break
        return self.dictionary.ids_to_words(result)
