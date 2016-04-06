#!/bin/bash -e

project_dir=$(readlink -f $(dirname "$0"))

export PYTHONPATH="${project_dir}:${PYTHONPATH}"
"${project_dir}/theanolm/tests/testvocabulary.py"
"${project_dir}/theanolm/tests/testiterators.py"
"${project_dir}/theanolm/tests/testtrainers.py"
"${project_dir}/wordclasses/tests/teststatistics.py"
"${project_dir}/wordclasses/tests/testbigramoptimizer.py"
