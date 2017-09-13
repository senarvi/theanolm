#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the "theanolm version" command.
"""

import theano
import pygpu

from theanolm import __version__

def version(args):
    """A function that performs the "theanolm version" command.

    :type args: argparse.Namespace
    :param args: a collection of command line arguments
    """

    print("TheanoLM", __version__)
    print("Theano", theano.version.version)
    try:
        pygpu_versions = pygpu._version.get_versions()
        print("pygpu {} (revision {})"
              .format(pygpu_versions["version"],
                      pygpu_versions["full-revisionid"]))
    except AttributeError:
        print("Old pygpu? Unable to get version information.")
