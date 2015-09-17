#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from exceptions import InvalidInput

class Dictionary(object):
	"""Word Dictionary
	
	Dictionary class provides a mapping between the words and word or class IDs.
	"""
	
	class WordClass(object):
		"""Collection of Words and Their Membership Probabilities
		
		A word class contains one or more words and their probabilities within
		the class. When word classes are not used, every class contains exactly
		one word with probability 1.
		"""
		
		def __init__(self, word):
			"""Initializes the class with one word with probability 1.
			
			:type word: string
			:param word: the initial word in the class
			"""
		
			self.id = None
			self._probs = {word: 1.0}
		
		def add(self, word, prob=1.0):
			"""Adds a word to the class with given probability.
			
			The membership probabilities are not guaranteed to be normalized.
			
			:type word: string
			:param word: the word to add to the class
			
			:type prob: float
			:param prob: the membership probability of the new word
			"""
			
			self._probs[word] = prob
		
		def normalize_probs(self):
			"""Normalizes the class membership probabilities to sum to one.
			
			:type word: string
			:param word: the word to add to the class
			
			:type prob: float
			:param prob: the membership probability of the new word
			"""
			
			prob_sum = sum(self._probs.values())
			for word in self._probs:
				self._probs[word] /= prob_sum
		
		def sample(self):
			"""Samples a word from the membership probability distribution.
			
			:rtype: str
			:returns: a random word from this class
			"""
			
			return next(iter(self._probs.keys()))
	
	def __init__(self, input_file):
		"""Creates word classes.
		
		:type input_file: file object
		:param input_file: input dictionary file
		"""
		
		# The word classes with consecutive indices. The first two classes are
		# the sentence break and the unknown word token. 
		self._word_classes = [
				Dictionary.WordClass('<sb>'),
				Dictionary.WordClass('<UNK>')]
		# Mapping from the IDs in the file to our word classes.
		file_id_to_class = dict()
		# Mapping from word strings to word classes.
		self._word_to_class = {
				'<sb>': self._word_classes[0],
				'<UNK>': self._word_classes[1]}
		self.sb_id = 0
		self.unk_id = 1
		
		for line in input_file:
			line = line.strip()
			fields = line.split()
			if len(fields) == 0:
				continue
			elif len(fields) == 1:
				word = fields[0]
				file_id = None
			elif len(fields) == 2:
				word = fields[0]
				file_id = fields[1]
			else:
				raise InvalidInput("%d fields on one line of dictionary file: %s" % (len(fields), line))
			
			if word in self._word_to_class:
				raise InvalidInput("Word `%s' appears more than once in the dictionary file." % word)
			if file_id in file_id_to_class:
				word_class = file_id_to_class[file_id]
				word_class.add(word)
			else:
				# No ID in the file or a new ID.
				word_class = Dictionary.WordClass(word)
				self._word_classes.append(word_class)
				if not file_id is None:
					file_id_to_class[file_id] = word_class
			self._word_to_class[word] = word_class
		
		for i, word_class in enumerate(self._word_classes):
			word_class.id = i
			word_class.normalize_probs()

	def num_words(self):
		"""Returns the number of words in the dictionary.
		
		:rtype: int
		:returns: the number of words in the dictionary
		"""
		
		return len(self._word_to_class)

	def num_classes(self):
		"""Returns the number of word classes.
		
		:rtype: int
		:returns: the number of words classes
		"""
		
		return len(self._word_classes)

	def text_to_ids(self, text):
		"""Translates words into word (class) IDs.
		
		:type text: list of strings
		:param text: a list of words
		
		:rtype: list of ints
		:returns: the given words translated into word IDs
		"""
		
		return [self._word_to_class[word].id if word in self._word_to_class else self.unk_id
				for word in text]

	def ids_to_text(self, word_ids):
		"""Translates word (class) IDs into words. If classes are used, samples
		a word from the membership probability distribution.
		
		:type text: list of ints
		:param text: a list of word IDs
		
		:rtype: list of strings
		:returns: the given word IDs translated into words
		"""
		
		return [self._word_classes[word_id].sample() for word_id in word_ids]
