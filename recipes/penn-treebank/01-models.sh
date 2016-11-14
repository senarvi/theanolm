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

# Set common training parameters.
OPTIMIZATION_METHOD=sgd
MAX_GRADIENT_NORM=5
STOPPING_CRITERION=no-improvement
VALIDATION_FREQ=1
PATIENCE=0

### softmax ####################################################################

#ARCHITECTURE_FILE="${arch_dir}/word-lstm256.arch"
#COST=cross-entropy
#LEARNING_RATE=10
#rm -f "${OUTPUT_DIR}/nnlm.h5"
#train
#compute_perplexity

# Model performance stopped improving. Decreasing learning rate from 1.25 to 0.625 and resetting state to 100 % of epoch 9.
# Finished training epoch 9. Best validation perplexity 121.57.
# Best validation set perplexity: 121.557743029
# Number of sentences: 3761
# Number of words: 200582
# Number of predicted probabilities: 82430
# Cross entropy (base e): 4.760321860161241
# Perplexity: 116.78350780925942
# ./01-models.sh  3878.42s user 1433.86s system 99% cpu 1:28:35.36 total

### hierarchical softmax #######################################################

#ARCHITECTURE_FILE="${arch_dir}/word-lstm256-hsoftmax.arch"
#COST=cross-entropy
#LEARNING_RATE=10
#rm -f "${OUTPUT_DIR}/nnlm.h5"
#train
#compute_perplexity

# Model performance stopped improving. Decreasing learning rate from 2.5 to 1.25 and resetting state to 100 % of epoch 8.
# Finished training epoch 8. Best validation perplexity 131.90.
# Best validation set perplexity: 131.79806381
# Number of sentences: 3761
# Number of words: 200582
# Number of predicted probabilities: 82430
# Cross entropy (base e): 4.836193322481491
# Perplexity: 125.98883885114117

## noise-contrastive estimation ################################################

#ARCHITECTURE_FILE="${arch_dir}/word-lstm256.arch"
#COST=nce
#NUM_NOISE_SAMPLES=3
#LEARNING_RATE=25
#rm -f "${OUTPUT_DIR}/nnlm.h5"
#train
#compute_perplexity

# Model performance stopped improving. Decreasing learning rate from 0.390625 to 0.1953125 and resetting state to 100 % of epoch 16.
# Finished training epoch 16. Best validation perplexity 149.60.
# Best validation set perplexity: 149.583726392
# Number of sentences: 3761
# Number of words: 200582
# Number of predicted probabilities: 82430
# Cross entropy (base e): 4.979556777848972
# Perplexity: 145.40991847206436
# ./01-models.sh  9865.74s user 2774.36s system 100% cpu 3:30:39.85 total

ARCHITECTURE_FILE="${arch_dir}/word-lstm256.arch"
COST=nce
NUM_NOISE_SAMPLES=5
LEARNING_RATE=25
rm -f "${OUTPUT_DIR}/nnlm.h5"
train
compute_perplexity

#ARCHITECTURE_FILE="${arch_dir}/word-lstm256.arch"
#COST=nce
#NUM_NOISE_SAMPLES=10
#LEARNING_RATE=25
#rm -f "${OUTPUT_DIR}/nnlm.h5"
#train
#compute_perplexity

# Model performance stopped improving. Decreasing learning rate from 0.09765625 to 0.048828125 and resetting state to 100 % of epoch 17.
# Finished training epoch 17. Best validation perplexity 181.93.
# Best validation set perplexity: 181.91759066
# Number of sentences: 3761
# Number of words: 86191
# Number of predicted probabilities: 82430
# Cross entropy (base e): 5.151167378858609
# Perplexity: 172.63290073146558
# ./01-models.sh  6643.96s user 2195.04s system 99% cpu 2:27:19.72 total

## noise-contrastive estimation with shared noise samples ######################

#ARCHITECTURE_FILE="${arch_dir}/word-lstm256.arch"
#COST=nce-shared
#NUM_NOISE_SAMPLES=100
#LEARNING_RATE=500
#rm -f "${OUTPUT_DIR}/nnlm.h5"
#train
#compute_perplexity

# Model performance stopped improving. Decreasing learning rate from 0.9765625 to 0.48828125 and resetting state to 100 % of epoch 16.
# Finished training epoch 16. Best validation perplexity 189.13.
# Training finished.
# Best validation set perplexity: 189.122505233
# train finished.
# Number of sentences: 3761
# Number of words: 86191
# Number of predicted probabilities: 82430
# Cross entropy (base e): 5.198794576222683
# Perplexity: 181.05386365022255
# ./01-models.sh  5243.30s user 2135.05s system 99% cpu 2:03:02.97 total
