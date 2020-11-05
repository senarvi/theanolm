#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# A Python class that stores word class definitions.

class WordClasses:
	class WordClass:
		def __init__(self, name):
      """
      Initialize a new instance.

      Args:
          self: (todo): write your description
          name: (str): write your description
      """
			self.__name = name
			self.__probs = dict()
		
		def __contains__(self, word):
      """
      Returns true if word is contained in word.

      Args:
          self: (todo): write your description
          word: (str): write your description
      """
			return word in self.__probs
		
		def __iter__(self):
      """
      Iterate over all words.

      Args:
          self: (todo): write your description
      """
			for word, prob in self.__probs.items():
				yield word, prob
		
		def name(self):
      """
      The name of the name

      Args:
          self: (todo): write your description
      """
			return self.__name
		
		def add(self, word, prob):
      """
      Add a word to the model.

      Args:
          self: (todo): write your description
          word: (str): write your description
          prob: (int): write your description
      """
			self.__probs[word] = float(prob)
		
		def get_probability(self, word):
      """
      Returns the probability of a word.

      Args:
          self: (todo): write your description
          word: (str): write your description
      """
			return self.__probs[word]
		
		def set_probability(self, word, prob):
      """
      Set the probability for a probability.

      Args:
          self: (todo): write your description
          word: (str): write your description
          prob: (todo): write your description
      """
			if not word in self.__probs:
				raise ValueError("WordClasses.set_probability: No word " + word + " in class " + self.__name + ".")
			self.__probs[word] = prob
		
		def normalize(self):
      """
      Normalizes the probability.

      Args:
          self: (todo): write your description
      """
			total_prob = sum(self.__probs.values())
			if total_prob > 0:
				factor = 1 / total_prob
			else:
				factor = 0
			for word, prob in self.__probs.items():
				self.__probs[word] = prob * factor
		
		def write(self, output_file):
      """
      Writes out to a file.

      Args:
          self: (todo): write your description
          output_file: (todo): write your description
      """
			for word, prob in self.__probs.items():
				output_file.write(self.__name)
				output_file.write(' ')
				output_file.write(str(prob))
				output_file.write(' ')
				output_file.write(word)
				output_file.write('\n')
	
	def __init__(self):
     """
     Initialize the internal state.

     Args:
         self: (todo): write your description
     """
		self.__classes = dict()
		self.__next_id = 1
	
	def __iter__(self):
     """
     Iterate over all classes of the class ).

     Args:
         self: (todo): write your description
     """
		for name, cls in self.__classes.items():
			yield cls
	
	def read(self, input_file):
     """
     Read a text file.

     Args:
         self: (todo): write your description
         input_file: (str): write your description
     """
		for line in input_file:
			fields = line.split()
			if len(fields) == 0:
				continue
			elif len(fields) == 2:
				# Assume mkcls / word2vec file format.
				word = fields[0]
				name = "CLASS-" + fields[1].zfill(5)
				prob = 0
			elif len(fields) == 3:
				# Assume SRILM file format.
				name = fields[0]
				prob = float(fields[1])
				word = fields[2]
			else:
				raise Exception("Invalid word class definition: " + line)
			if not name in self.__classes:
				self.__classes[name] = self.WordClass(name)
			self.__classes[name].add(word, prob)

	def write(self, output_file):
     """
     Writes the model classes to the output file.

     Args:
         self: (todo): write your description
         output_file: (str): write your description
     """
		for name, cls in self.__classes.items():
			cls.normalize()
			cls.write(output_file)
	
	def create(self, name=None):
     """
     Creates a new class.

     Args:
         self: (int): write your description
         name: (str): write your description
     """
		if name is None:
			while True:
				name = 'CLASS-' + ('%05d' % self.__next_id)
				self.__next_id += 1
				if not name in self.__classes.keys():
					break
		new_class = self.WordClass(name)
		self.__classes[name] = new_class
		return new_class
	
	def find_containing(self, word):
     """
     Return the first matching word.

     Args:
         self: (todo): write your description
         word: (str): write your description
     """
		for name, cls in self.__classes.items():
			if word in cls:
				return cls
		return None

class WordsToClasses:
	def __init__(self, word_classes):
     """
     Initialize the class from a list of classes.

     Args:
         self: (todo): write your description
         word_classes: (str): write your description
     """
		self.__map = dict()
		for cls in word_classes:
			for word, prob in cls:
				self.__map[word] = cls.name()
	
	def __getitem__(self, word):
     """
     Returns the item from the word.

     Args:
         self: (todo): write your description
         word: (todo): write your description
     """
		return self.__map[word]
	
	def __contains__(self, word):
     """
     Determine if word is contained in word.

     Args:
         self: (todo): write your description
         word: (str): write your description
     """
		return word in self.__map
