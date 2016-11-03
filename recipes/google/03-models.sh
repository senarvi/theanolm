#!/bin/bash -e

script_dir=$(dirname "${0}")
script_dir=$(readlink -e "${script_dir}")
arch_dir="${script_dir}/../architectures"

# Load paths to the corpus files. You need to create paths.sh with:
#
# TRAIN_FILES=(/path/to/google/1-billion-word-language-modeling-benchmark/training-monolingual.tokenized.shuffled/news.en-*-of-*)
# DEVEL_FILE=/path/to/google/1-billion-word-language-modeling-benchmark/heldout-monolingual.tokenized.shuffled/news.en.heldout-00000-of-00050
# EVAL_FILE=/path/to/google/1-billion-word-language-modeling-benchmark/heldout-monolingual.tokenized.shuffled/news.en.heldout-00001-of-00050
# OUTPUT_DIR=/path/to/output/directory
#
source "${script_dir}/paths.sh"

# Load common functions.
source "${script_dir}/../common/functions.sh"

# Set common training parameters.
MAX_GRADIENT_NORM=5
STOPPING_CRITERION=no-improvement

### classes, LSTM 256 ##########################################################

ARCHITECTURE_FILE="${arch_dir}/class-lstm256.arch"
CLASSES="${OUTPUT_DIR}/classes"
COST=cross-entropy
OPTIMIZATION_METHOD=sgd
LEARNING_RATE=1.0
rm -f "${OUTPUT_DIR}/nnlm.h5"
train
compute_perplexity
