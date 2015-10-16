#!/bin/sh

project_dir=$(readlink -f $(dirname "$0"))
tests_dir="${project_dir}/theanolm/tests"

export PYTHONPATH="${project_dir}:${PYTHONPATH}"
"${tests_dir}/testiterators.py"
"${tests_dir}/testtrainers.py"
