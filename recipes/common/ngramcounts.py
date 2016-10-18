#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# A Python class that stores n-gram counts.

import sys

class NGramCounts:
	def __init__(self):
		self.__counts = dict()
	
	def __contains__(self, ngram):
		return ngram in self.__counts
	
	def __getitem__(self, ngram):
		return self.__counts[ngram]
	
	def read(self, input_file, max_order=None, min_count=None):
		lines_read = 0
		for line in input_file:
			lines_read += 1
			if lines_read % 10000000 == 0:
				print(lines_read, "lines read.", file=sys.stderr)
			tab_pos = line.find('\t')
			ngram = tuple(line[:tab_pos].split())
			if max_order is not None and len(ngram) > max_order:
				continue
			count = int(line[tab_pos+1:])
			if min_count is not None and count < min_count:
				continue
			self.__counts[ngram] = count
	
	def from_text(self, text, max_order):
		lines = text.splitlines()
		history = []
		for line in lines:
			words = line.split()
			for word in words:
				history.append(word)
				if len(history) > max_order:
					history = history[1:]
				for i in range(min(max_order, len(history))):
					self.increment(history[-i-1:])

	def write(self, output_file):
		for ngram, count in self.__counts.items():
			output_file.write(' '.join(ngram) + '\t' + str(count) + '\n')
	
	def increment(self, ngram):
		ngram = tuple(ngram)
		if ngram in self.__counts:
			self.__counts[ngram] += 1
		else:
			self.__counts[ngram] = 1

	def level(self, n):
		for ngram, count in self.__counts.items():
			if len(ngram) == n:
				yield ngram
	
	# Prune n-grams with count less than or equal to n.
	def prune_count(self, n):
		prune = [ngram for ngram, count in self.__counts.items() if count <= n]
		for ngram in prune:
			del mydict[ngram]

	def num_contained(self, other):
		result = 0
		for ngram in self.__counts:
			if ngram in other:
				result += 1
		return result
	
	def num_ngrams(self):
		return len(self.__counts)
