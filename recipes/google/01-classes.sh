#!/bin/bash -e

script_dir=$(dirname "${0}")
script_dir=$(readlink -e "${script_dir}")

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

# Specify parameters for word class optimization.
NUM_CLASSES="5000"
VOCAB_MIN_COUNT="3"

create_classes
# log likelihood: -5016691478
# Train run time: 430015 seconds
# create_classes finished.
# ./01-classes.sh  1444672.12s user 1390.50s system 334% cpu 120:09:32.07 total
