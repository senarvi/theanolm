#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess

def version(args):
    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        git_description = subprocess.check_output(['git', 'describe'],
                                                  cwd=script_dir)
        git_description = git_description.decode('utf-8').rstrip()
        print("TheanoLM", git_description)
    except subprocess.CalledProcessError:
        print("Git repository description is not available.")
