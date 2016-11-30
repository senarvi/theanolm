#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def utterance_from_line(line):
    """Converts a line of text, read from an input file, into a list of words.

    Start-of-sentence and end-of-sentece tags (``<s>`` and ``</s>``) will be
    inserted at the beginning and the end of the list, if they're missing. If
    the line is empty, returns an empty list (instead of an empty sentence
    ``['<s>', '</s>']``).

    :type line: str or bytes
    :param line: a line of text (read from an input file)
    """

    if type(line) == bytes:
        line = line.decode('utf-8')
    line = line.rstrip()
    if not line:
        # empty line
        return []

    result = line.split()
    if result[0] != '<s>':
        result.insert(0, '<s>')
    if result[-1] != '</s>':
        result.append('</s>')

    return result

def find_sentence_starts(data):
    """Finds the positions inside a memory-mapped file, where the sentences
    (lines) start.

    TextIOWrapper disables tell() when readline() is called, so search for
    sentence starts in memory-mapped data.

    :type data: mmap.mmap
    :param data: memory-mapped data of the input file

    :rtype: list of ints
    :returns: a list of file offsets pointing to the next character from a
              newline (including file start and excluding file end)
    """

    result = [0]

    pos = 0
    while True:
        pos = data.find(b'\n', pos)
        if pos == -1:
            break;
        pos += 1
        if pos < len(data):
            result.append(pos)

    return result
