#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import subprocess
import numpy
import theano
import theanolm
from filetypes import TextFileType

def score_text(input_file, dictionary, scorer, output_file):
    validation_iter = theanolm.BatchIterator(
        input_file,
        dictionary,
        batch_size=1,
        max_sequence_length=None)

    total_logprob = 0
    num_words = 0
    num_sentences = 0
    for word_ids, membership_probs, mask in validation_iter:
        logprobs = scorer.score_batch(word_ids, membership_probs, mask)
        for seq_index, seq_logprobs in enumerate(logprobs):
            seq_logprob = sum(seq_logprobs)
            seq_length = len(seq_logprobs)
            total_logprob += seq_logprob
            num_words += seq_length
            num_sentences += 1
            if not args.verbose:
                continue
            seq_word_ids = word_ids[:, seq_index]
            args.output_file.write("### Sentence {0}\n".format(num_sentences))
            seq_details = [str(word_id) + ":" + str(logprob)
                for word_id, logprob in zip(seq_word_ids, seq_logprobs)]
            args.output_file.write(" ".join(seq_details) + "\n")
            args.output_file.write("Sentence perplexity: {0}\n\n".format(
                numpy.exp(-seq_logprob / seq_length)))

    output_file.write("Number of words: {0}\n".format(num_words))
    output_file.write("Number of sentences: {0}\n".format(num_sentences))
    if num_words > 0:
        cross_entropy = -total_logprob / num_words
        perplexity_e = numpy.exp(cross_entropy)
        perplexity_10 = numpy.power(10, cross_entropy)
        output_file.write("Cross entropy: {0}\n".format(cross_entropy))
        output_file.write("Perplexity (base e): {0}\n".format(perplexity_e))
        output_file.write("Perplexity (base 10): {0}\n".format(perplexity_10))


def rescore_nbest(input_file, dictionary, scorer, output_file, lscore_field=1,
                  w1_field=3):
    for line_num, line in enumerate(input_file):
        fields = line.split()
        
        words = fields[w1_field:]
        words.append('<sb>')
        
        word_ids = dictionary.words_to_ids(words)
        word_ids = numpy.array([[x] for x in word_ids]).astype('int64')
        
        probs = dictionary.words_to_probs(words)
        probs = numpy.array([[x] for x in probs]).astype(theano.config.floatX)

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

argument_group = parser.add_argument_group("scoring")
argument_group.add_argument(
    '--verbose', action='store_true',
    help='print detailed per word probabilities')

args = parser.parse_args()

try:
    script_path = os.path.dirname(os.path.realpath(__file__))
    git_description = subprocess.check_output(['git', 'describe'], cwd=script_path)
    print("TheanoLM %s", git_description.decode('utf-8'))
except subprocess.CalledProcessError:
    pass

print("Reading model state from %s." % args.model_path)
state = numpy.load(args.model_path)

print("Reading dictionary.")
sys.stdout.flush()
dictionary = theanolm.Dictionary(args.dictionary_file, args.dictionary_format)
print("Number of words in vocabulary:", dictionary.num_words())
print("Number of word classes:", dictionary.num_classes())

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
    score_text(args.input_file, dictionary, scorer, args.output_file)
elif args.input_format == 'srilm-nbest':
    print("Rescoring n-best list.")
    rescore_nbest(args.input_file, dictionary, scorer, args.output_file)
elif args.input_format == 'id-nbest':
    print("Rescoring n-best list.")
    rescore_nbest(args.input_file, dictionary, scorer, args.output_file,
                  lscore_field=2, w1_field=4)
