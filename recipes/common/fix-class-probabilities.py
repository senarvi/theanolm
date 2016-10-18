#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Reads a class definitions file and an n-gram counts file, and corrects the
# class expansion probabilities according to the unigram counts of the words.

import argparse
import sys
from filetypes import TextFileType
from wordclasses import WordClasses, WordsToClasses
from ngramcounts import NGramCounts

parser = argparse.ArgumentParser()
parser.add_argument('classes', type=TextFileType('r'), help='input class definitions file')
parser.add_argument('counts', type=TextFileType('r'), help='n-gram counts file')
args = parser.parse_args()

classes = WordClasses()
classes.read(args.classes)

word_counts = NGramCounts()
word_counts.read(args.counts)

for cls in classes:
	counts = dict()
	for word, prob in cls:
		unigram = tuple([word])
		if unigram in word_counts:
			counts[word] = word_counts[unigram]
		else:
			counts[word] = 0
	total_count = sum(counts.values())
	if total_count != 0:
		for word, count in counts.items():
			cls.set_probability(word, float(count) / total_count)

classes.write(sys.stdout)
