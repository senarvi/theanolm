#!/bin/sh

SRC_ROOT=$(dirname "$0")

export PYTHONPATH="$SRC_ROOT:$PYTHONPATH"
"$SRC_ROOT/tests/testiterators.py"
