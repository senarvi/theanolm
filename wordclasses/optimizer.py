#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy

class Optimizer(object):
    '''
    Clusters words into classes.
    '''

    def __init__(self, num_classes, corpus_file, vocabulary_file = None):
        '''
        Reads the vocabulary from the text ``corpus_file``. The vocabulary may
        be restricted by ``vocabulary_file``. Then reads the statistics from the
        text.
        '''

        # Read the vocabulary.
        self.vocabulary = set(['<s>', '</s>', '<UNK>'])
        if vocabulary_file is None:
            for line in corpus_file:
                self.vocabulary.update(line.split())
        else:
            restrict_words = set(line.strip() for line in vocabulary_file)
            for line in corpus_file:
                for word in line.split():
                    if word in restrict_words:
                        self.vocabulary.add(word)

        # Create word IDs and read word statistics.
        self.vocabulary_size = len(self.vocabulary)
        self.word_ids = dict(zip(self.vocabulary, range(self.vocabulary_size)))
        corpus_file.seek(0)
        self._read_word_statistics(corpus_file)

        # Initialize classes and compute class statistics.
        self._random_init_classes(num_classes)
        self._compute_class_statistics()

    def _read_word_statistics(self, corpus_file):
        '''
        Reads word statistics from corpus file.
        '''

        self.word_counts = numpy.zeros(self.vocabulary_size, numpy.int64)
        self.word_word_counts = numpy.zeros(
            (self.vocabulary_size, self.vocabulary_size), numpy.int64)

        for line in corpus_file:
            sentence = [self.word_ids['<s>']]
            for word in line.split():
                if word in self.vocabulary:
                    sentence.append(self.word_ids[word])
                else:
                    sentence.append(self.word_ids['<UNK>'])
            sentence.append(self.word_ids['</s>'])
            for word_id in sentence:
                self.word_counts[word_id] += 1
            for left_word_id, right_word_id in zip(sentence[:-1], sentence[1:]):
                self.word_word_counts[left_word_id, right_word_id] += 1

    def _random_init_classes(self, num_classes):
        '''
        Randomly initialize word classes.
        '''
        
        self.num_classes = num_classes + 3

        self.word_to_class = [None] * self.vocabulary_size
        self.class_to_words = [set() for _ in range(self.num_classes)]

        self.word_to_class[self.word_ids['<s>']] = 0
        self.class_to_words[0].add(self.word_ids['<s>'])
        self.word_to_class[self.word_ids['</s>']] = 1
        self.class_to_words[1].add(self.word_ids['</s>'])
        self.word_to_class[self.word_ids['<UNK>']] = 2
        self.class_to_words[2].add(self.word_ids['<UNK>'])

        class_id = 3
        for word_id, _ in sorted(enumerate(self.word_counts),
                                 key=lambda x: x[1]):
            if not self.word_to_class[word_id] is None:
                # A class has been already assigned to <s>, </s>, and <UNK>.
                continue
            self.word_to_class[word_id] = class_id
            self.class_to_words[class_id].add(word_id)
            class_id = min((class_id + 1) % self.num_classes, 3)

    def _compute_class_statistics(self):
        '''
        Computes class statistics.
        '''

        self.class_counts = numpy.zeros(self.num_classes, numpy.int64)
        self.class_class_counts = numpy.zeros(
            (self.num_classes, self.num_classes), numpy.int64)
        self.class_word_counts = numpy.zeros(
            (self.num_classes, self.vocabulary_size), numpy.int64)
        self.word_class_counts = numpy.zeros(
            (self.vocabulary_size, self.num_classes), numpy.int64)

        for word_id, class_id in enumerate(self.word_to_class):
            self.class_counts[class_id] += self.word_counts[word_id]
        for (left_word_id, right_word_id), count in numpy.ndenumerate(self.word_word_counts):
            left_class_id = self.word_to_class[left_word_id]
            right_class_id = self.word_to_class[right_word_id]
            self.class_class_counts[left_class_id,right_class_id] += count
            self.class_word_counts[left_class_id,right_word_id] += count
            self.word_class_counts[left_word_id,right_class_id] += count

    def _evaluate_move(self, word_id, new_class_id):
        '''
        Evaluates how much moving a word to another class would change the log
        likelihood.
        '''

        pass

    def _move(self, word_id, new_class_id):
        '''
        Moves a word to another class.
        '''

        old_class_id = self.word_to_class[word_id]
        word_count = self.word_counts[word_id]
        self.class_counts[old_class_id] -= word_count
        self.class_counts[new_class_id] += word_count
        
        for right_word_id, count in enumerate(self.word_word_counts[word_id,:]):
            if right_word_id == word_id:
                continue
            right_class_id = self.word_to_class[right_word_id]
            self.class_class_counts[old_class_id,right_class_id] -= count
            self.class_class_counts[new_class_id,right_class_id] += count
            self.class_word_counts[old_class_id,right_word_id] -= count
            self.class_word_counts[new_class_id,right_word_id] += count
        
        for left_word_id, count in enumerate(self.word_word_counts[:,word_id]):
            if left_word_id == word_id:
                continue
            left_class_id = self.word_to_class[left_word_id]
            self.class_class_counts[left_class_id,old_class_id] -= count
            self.class_class_counts[left_class_id,new_class_id] += count
            self.word_class_counts[left_word_id,old_class_id] -= count
            self.word_class_counts[left_word_id,new_class_id] += count

        count = self.word_word_counts[word_id,word_id]
        self.class_class_counts[old_class_id,old_class_id] -= count
        self.class_class_counts[new_class_id,new_class_id] += count
        self.class_word_counts[old_class_id,word_id] -= count
        self.class_word_counts[new_class_id,word_id] += count
        self.word_class_counts[word_id,old_class_id] -= count
        self.word_class_counts[word_id,new_class_id] += count

        self.class_to_words[old_class_id].remove(word_id)
        self.class_to_words[new_class_id].add(word_id)
        self.word_to_class[word_id] = new_class_id
