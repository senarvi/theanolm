#!/bin/bash -e

TRAIN_FILES=("${GROUP_DIR}"/c/penn-treebank-project/ptb.train.txt)
DEVEL_FILE="${GROUP_DIR}"/c/penn-treebank-project/ptb.valid.txt
EVAL_FILE="${GROUP_DIR}"/c/penn-treebank-project/ptb.test.txt
OUTPUT_DIR="${WORK_DIR}"/theanolm-recipes/penn-treebank

module purge
module load srilm
export PYTHONPATH="${PYTHONPATH}:${HOME}/git/theanolm"
export PATH="${PATH}:${HOME}/git/theanolm/bin"
