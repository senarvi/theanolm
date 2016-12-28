#!/bin/bash -e
#
# Examples for training TheanoLM models on Penn Treebank corpus. The results (in
# comments) have been obtained using the processed data that is distributed with
# RNNLM basic examples. The vocabulary is 10002 words including the <s> and </s>
# symbols. With such a small vocabulary, noise-contrastive estimation does not
# improve training speed. Hierarchical softmax improves training speed with only
# a small degradation in model performance.

script_dir=$(dirname "${0}")
script_dir=$(readlink -e "${script_dir}")
arch_dir="${script_dir}/../architectures"

# Load paths to the corpus files. You need to download the Penn Treebank corpus
# e.g. from http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz and
# create paths.sh with:
#
# TRAIN_FILES=(/path/to/penn-treebank-project/ptb.train.txt)
# DEVEL_FILE=/path/to/penn-treebank-project/ptb.valid.txt
# EVAL_FILE=/path/to/penn-treebank-project/ptb.test.txt
# OUTPUT_DIR=/path/to/output/directory
#
source "${script_dir}/paths.sh"

# Load common functions.
source "${script_dir}/../common/functions.sh"
source "${script_dir}/../common/configure-theano.sh"

# Set training parameters.
BATCH_SIZE=24
OPTIMIZATION_METHOD=sgd
MAX_GRADIENT_NORM=5
STOPPING_CRITERION=no-improvement
VALIDATION_FREQ=1
PATIENCE=0
ARCHITECTURE_FILE="${arch_dir}/word-lstm256.arch"
COST=nce
NUM_NOISE_SAMPLES=100
NOISE_SHARING=batch
NOISE_DAMPENING=0.75
LEARNING_RATE=5
#PROFILE=1

rm -f "${OUTPUT_DIR}/nnlm.h5"
mv -f "${script_dir}/nce-shared.log" "${script_dir}/nce-shared.log~" 2>/dev/null || true
train | tee "${script_dir}/nce-shared.log"
compute_perplexity | tee --append "${script_dir}/nce-shared.log"
