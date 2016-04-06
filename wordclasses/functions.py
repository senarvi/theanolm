#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def byte_size(x):
    """Converts a byte size into a human-readable string.

    :type x: int
    :param x: number of bytes

    :rtype: str
    :returns: the given size formatted in a human-readable string
    """

    suffixes = ['bytes', 'KB', 'MB', 'GB', 'TB']
    index = 0
    while x > 1024 and index < 4:
        index += 1
        x /= 1024
    return "{} {}".format(int(round(x)), suffixes[index])
