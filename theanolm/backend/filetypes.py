#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that defines file type classes to be used with command-line argument
parser.
"""

import io
import sys
import argparse
import gzip

class TextFileType(object):
    """An object that can be passed as the "type" argument to
    ArgumentParser.add_argument() in order to convert a path argument to a
    file object.

    If the path ends in ".gz", the file will be opened using gzip.open().
    UTF-8 encoding will be assumed. The special path "-" means standard
    input or output.

    Keyword Arguments:
      - mode -- A string indicating how the file is to be opened. Accepts the
        same values as the builtin open() function.
    """

    def __init__(self, mode='r'):
        """
        Initialize a mode.

        Args:
            self: (todo): write your description
            mode: (todo): write your description
        """
        self._mode = mode

    def __call__(self, string):
        """
        Call the given string.

        Args:
            self: (todo): write your description
            string: (str): write your description
        """
        if string is None:
            return None

        if string == '-':
            if 'r' in self._mode:
                return io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
            elif ('w' in self._mode) or ('a' in self._mode):
                return io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            else:
                msg = "Cannot use standard input or output (file is opened " \
                      "with mode '%r')." % self._mode
                raise argparse.ArgumentTypeError(msg)

        try:
            if string.endswith('.gz'):
                return gzip.open(string, self._mode + 't', encoding='utf-8')
            return open(string, self._mode + 't', encoding='utf-8')
        except IOError as e:
            message = "Cannot open '%s': %s" % (string, e)
            raise argparse.ArgumentTypeError(message)
        except Exception as e:
            # I don't know a better way to communicate exceptions to the user.
            raise argparse.ArgumentTypeError(str(e))

    def __repr__(self):
        """
        Return a repr representation of this object.

        Args:
            self: (todo): write your description
        """
        return '%s(%s)' % (type(self).__name__, self._mode)

class BinaryFileType(object):
    """An object that can be passed as the "type" argument to
    ArgumentParser.add_argument() in order to convert a path argument to a
    file object.

    If the path ends in ".gz", the file will be opened using gzip.open().
    UTF-8 encoding will be assumed. The special path "-" means standard
    input or output.

    Keyword Arguments:
      - mode -- A string indicating how the file is to be opened. Accepts the
        same values as the builtin open() function.
    """

    def __init__(self, mode='r'):
        """
        Initialize a mode.

        Args:
            self: (todo): write your description
            mode: (todo): write your description
        """
        self._mode = mode

    def __call__(self, string):
        """
        Call the given string.

        Args:
            self: (todo): write your description
            string: (str): write your description
        """
        if string is None:
            return None

        if string == '-':
            if 'r' in self._mode:
                return sys.stdin.buffer
            elif ('w' in self._mode) or ('a' in self._mode):
                return sys.stdout.buffer
            else:
                msg = "Cannot use standard input or output (file is opened " \
                      "with mode '%r')." % self._mode
                raise argparse.ArgumentTypeError(msg)

        try:
            if string.endswith('.gz'):
                return gzip.open(string, self._mode + 'b')
            return open(string, self._mode + 'b',)
        except IOError as e:
            message = "Cannot open '%s': %s" % (string, e)
            raise argparse.ArgumentTypeError(message)
        except Exception as e:
            # I don't know a better way to communicate exceptions to the user.
            raise argparse.ArgumentTypeError(str(e))

    def __repr__(self):
        """
        Return a repr representation of this object.

        Args:
            self: (todo): write your description
        """
        return '%s(%s)' % (type(self).__name__, self._mode)
