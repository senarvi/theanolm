#!/bin/bash -e

#
# An example script for sampling text using a TheanoLM model.
#

script_dir=$(dirname "${0}")
script_dir=$(readlink -e "${script_dir}")

# Load OUTPUT_DIR.
source "${script_dir}/paths.sh"

# Load common functions.
source "${script_dir}/../common/functions.sh"
source "${script_dir}/../common/configure-theano.sh"

theanolm sample --num-sentences 10 "${OUTPUT_DIR}/nnlm.h5"
