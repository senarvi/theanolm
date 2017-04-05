#!/bin/bash -e
#
# An example script for scoring text using a TheanoLM model.

script_dir=$(dirname "${0}")
script_dir=$(readlink -e "${script_dir}")

# Load OUTPUT_DIR.
source "${script_dir}/paths.sh"

# Load common functions.
source "${script_dir}/../common/functions.sh"
source "${script_dir}/../common/configure-theano.sh"

text_file=$(mktemp)
echo 'in some other markets as well' >"${text_file}"
theanolm score "${OUTPUT_DIR}/nnlm.h5" "${text_file}" --output word-scores
rm -f "${text_file}"
