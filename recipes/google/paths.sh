#!/bin/bash -e

TRAIN_FILES=("${GROUP_DIR}"/c/google/1-billion-word-language-modeling-benchmark/training-monolingual.tokenized.shuffled/news.en-*-of-*)
DEVEL_FILE="${GROUP_DIR}"/c/google/1-billion-word-language-modeling-benchmark/heldout-monolingual.tokenized.shuffled/news.en.heldout-00000-of-00050
EVAL_FILE="${GROUP_DIR}"/c/google/1-billion-word-language-modeling-benchmark/heldout-monolingual.tokenized.shuffled/news.en.heldout-00001-of-00050
OUTPUT_DIR="${WORK_DIR}"/theanolm-recipes/google
