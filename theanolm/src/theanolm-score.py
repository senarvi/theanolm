#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy
import theano
import theanolm
from filetypes import TextFileType

def rescore_nbest(input_file, dictionary, scorer, output_file, lscore_field=1, w1_field=3):
    for line_num, line in enumerate(input_file):
        fields = line.split()
        
        words = fields[w1_field:]
        words.append('<sb>')
        
        word_ids = dictionary.words_to_ids(words)
        word_ids = numpy.array([[x] for x in word_ids]).astype('int64')
        
        probs = dictionary.words_to_probs(words)
        probs = numpy.array([[x] for x in probs]).astype('float32')

        lm_score = scorer.score_sentence(word_ids, probs)
        fields[lscore_field] = str(lm_score)
        output_file.write(' '.join(fields) + '\n')

        if line_num % 100 == 0:
            print("%d sentences rescored." % (line_num + 1))
        sys.stdout.flush()

parser = argparse.ArgumentParser()

argument_group = parser.add_argument_group("files")
argument_group.add_argument(
    'model_path', metavar='MODEL', type=str,
    help='path where the best model state will be saved in numpy .npz format')
argument_group.add_argument(
    'input_file', metavar='INPUT', type=TextFileType('r'),
    help='text or .gz file containing text or n-best list to be scored (one sentence per line)')
argument_group.add_argument(
    'dictionary_file', metavar='DICTIONARY', type=TextFileType('r'),
    help='text or .gz file containing word list (one word per line) or word to class ID mappings (word and ID per line)')
argument_group.add_argument(
    '--input-format', metavar='FORMAT', type=str, default='text',
    help='input text format, one of "text" (one sentence per line, default), "srilm-nbest" (n-best list containing "ascore lscore nwords w1 w2 w3 ..." on each line), "id-nbest" (n-best list containing "id ascore lscore nwords w1 w2 w3 ..." on each line)')
argument_group.add_argument(
    '--dictionary-format', metavar='FORMAT', type=str, default='words',
    help='dictionary format, one of "words" (one word per line, default), "classes" (word and class ID per line), "srilm-classes" (class name, membership probability, and word per line)')
argument_group.add_argument(
    '--output-file', metavar='OUTPUT', type=TextFileType('w'), default='-',
    help='where to write the score or rescored n-best list (default stdout)')
argument_group.add_argument(
    '--num-samples', dest='num_samples', type=int, default=0,
    help='Number of example sentences to generate')

args = parser.parse_args()

state = numpy.load(args.model_path)

print("Reading dictionary.")
sys.stdout.flush()
dictionary = theanolm.Dictionary(args.dictionary_file, args.dictionary_format)
print("Number of words in vocabulary:", dictionary.num_words())
print("Number of word classes:", dictionary.num_classes())

validation_iter = theanolm.BatchIterator(
    args.input_file,
    dictionary,
    batch_size=1,
    max_sequence_length=None)

print("Building neural network.")
sys.stdout.flush()
architecture = theanolm.Network.Architecture.from_state(state)
print(architecture)
network = theanolm.Network(dictionary, architecture)
print("Restoring neural network state.")
network.set_state(state)

print("Building text scorer.")
sys.stdout.flush()
scorer = theanolm.TextScorer(network)

if args.input_format == 'text':
    costs, counts = scorer.negative_log_probabilities(validation_iter)
    sentence_average = costs.mean()
    per_word = costs.sum() / counts.sum()
    perplexity_per_word = numpy.exp(-per_word)
    args.output_file.write(
        "Average sentence negative log probability: "
        "{}\n".format(sentence_average))
    args.output_file.write(
        "Average word negative log probability: "
        "{}\n".format(per_word))
    args.output_file.write(
        "Perplexity per word: {}\n".format(perplexity_per_word))
elif args.input_format == 'srilm-nbest':
    print("Rescoring n-best list.")
    rescore_nbest(args.input_file, dictionary, scorer, args.output_file)
elif args.input_format == 'id-nbest':
    print("Rescoring n-best list.")
    rescore_nbest(args.input_file, dictionary, scorer, args.output_file, lscore_field=2, w1_field=4)

if args.num_samples > 0:
    print("Sampling...")
    sampler = theanolm.TextSampler(rnnlm, dictionary)
    for i in range(args.num_samples):
        words = sampler.generate()
        args.output_file.write('{}: {}\n'.format(
            i, ' '.join(words)))
