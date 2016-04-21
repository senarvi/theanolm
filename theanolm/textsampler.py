#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import theano

class TextSampler(object):
    """Neural network language model sampler

    A Theano function that generates text using a neural network language
    model.
    """

    def __init__(self, network, vocabulary):
        """Creates the neural network architecture.

        Creates the function self.step_function that uses the state of the
        previous time step and the word ID of the current time step, to compute
        the output distribution. It samples from the output distribution and
        returns the sampled word ID along with the output state of this time
        step.

        :type network: Network
        :param network: the neural network object

        :type vocabulary: Vocabulary
        :param vocabulary: vocabulary that provides mapping between words and
                           word IDs
        """

        self.network = network
        self.vocabulary = vocabulary

        inputs = [self.network.word_input, self.network.class_input]
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

        sos_id = self.vocabulary.word_to_id['<s>']
        sos_class_id = self.vocabulary.word_id_to_class_id[sos_id]
        eos_id = self.vocabulary.word_to_id['</s>']

        # We are only generating one sequence at a time. The input is passed as
        # a 2-dimensional matrix with only one element, since in mini-batch mode
        # the matrix contains multiple sequences and time steps.
        result = [sos_id]
        word_input = sos_id * numpy.ones(shape=(1, 1)).astype('int64')
        class_input = sos_class_id * numpy.ones(shape=(1, 1)).astype('int64')

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
            step_result = self.step_function(word_input, class_input,
                                             *recurrent_state)
            class_ids = step_result[0]
            recurrent_state = step_result[1:]
            assert len(recurrent_state) == \
                   len(self.network.recurrent_state_size)
            # The class ID from the single time step from the single sequence.
            class_id = class_ids[0,0]
            word_id = self.vocabulary.class_id_to_word_id(class_id)
            result.append(word_id)
            if word_id == eos_id:
                break
            word_input = word_id * numpy.ones(shape=(1, 1)).astype('int64')
            class_input = class_ids
        return [self.vocabulary.id_to_word[word_id] for word_id in result]
