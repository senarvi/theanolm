#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Dictionary(object):
	"""Word Dictionary
	
	Dictionary class provides a mapping between the words and word IDs.
	"""
	
	def __init__(self, input_file):
		"""Creates a mapping from words to word IDs and vice versa.
		
		:type input_file: file object
		:param input_file: input dictionary file
		"""
		
		self.word_to_id = dict()
		self.id_to_word = dict()
		for i, line in enumerate(input_file):
			word = line.strip()
			word_id = i + 2
			self.id_to_word[word_id] = word
			self.word_to_id[word] = word_id

	def size(self):
		"""Returns the number of words in the dictionary.
		
		:rtype: int
		:returns: the number of words in the dictionary
		"""
		
		return len(self.word_to_id)

	def text_to_ids(self, text):
		"""Translates words into word IDs.
		
		:type text: list of strings
		:param text: a list of words
		
		:rtype: list of ints
		:returns: the given words translated into word IDs
		"""
		
		return [self.word_to_id[word] if word in self.word_to_id else 1
				for word in text]

	def ids_to_text(self, word_ids):
		"""Translates word IDs into words.
		
		:type text: list of ints
		:param text: a list of word IDs
		
		:rtype: list of strings
		:returns: the given word IDs translated into words
		"""
		
		return [self.id_to_word[word_id] if word_id in self.id_to_word else "UNK(%d)" % word_id
				for word_id in word_ids]
