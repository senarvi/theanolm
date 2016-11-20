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

# Model performance stopped improving. Decreasing learning rate from 1.25 to 0.625 and resetting state to 100 % of epoch 8.
# Finished training epoch 8. Best validation perplexity 121.18.
# Best validation set perplexity: 121.156465316
# Number of sentences: 3761
# Number of words: 200582
# Number of predicted probabilities: 82430
# Cross entropy (base e): 4.751490638853352
# Perplexity: 115.75670743075338
# ./01-models.sh  3538.38s user 1396.68s system 99% cpu 1:22:18.42 total

### hierarchical softmax #######################################################

#ARCHITECTURE_FILE="${arch_dir}/word-lstm256-hsoftmax.arch"
#COST=cross-entropy
#LEARNING_RATE=10
#rm -f "${OUTPUT_DIR}/nnlm.h5"
#train
#compute_perplexity

# Model performance stopped improving. Decreasing learning rate from 1.25 to 0.625 and resetting state to 100 % of epoch 8.
# Finished training epoch 8. Best validation perplexity 129.35.
# Best validation set perplexity: 129.327700525
# Number of sentences: 3761
# Number of words: 200582
# Number of predicted probabilities: 82430
# Cross entropy (base e): 4.8205827876949785
# Perplexity: 124.037357165272
# ./01-models.sh  2284.53s user 842.10s system 99% cpu 52:21.46 total

## noise-contrastive estimation ################################################

#ARCHITECTURE_FILE="${arch_dir}/word-lstm256.arch"
#COST=nce
#NUM_NOISE_SAMPLES=25
#LEARNING_RATE=20
#rm -f "${OUTPUT_DIR}/nnlm.h5"
#train
#compute_perplexity

# Model performance stopped improving. Decreasing learning rate from 0.15625 to 0.078125 and resetting state to 100 % of epoch 12.
# Finished training epoch 12. Best validation perplexity 168.38.
# Best validation set perplexity: 168.36442896
# Number of sentences: 3761
# Number of words: 200582
# Number of predicted probabilities: 82430
# Cross entropy (base e): 5.090273458248636
# Perplexity: 162.43427497302542
# ./01-models.sh  4617.24s user 1921.59s system 99% cpu 1:49:11.62 total

## noise-contrastive estimation with shared noise samples ######################

#ARCHITECTURE_FILE="${arch_dir}/word-lstm256.arch"
#COST=nce-shared
#NUM_NOISE_SAMPLES=100
#LEARNING_RATE=10
#rm -f "${OUTPUT_DIR}/nnlm.h5"
#train
#compute_perplexity

# Model performance stopped improving. Decreasing learning rate from 0.078125 to 0.0390625 and resetting state to 100 % of epoch 17.
# Finished training epoch 17. Best validation perplexity 166.00.
# Best validation set perplexity: 166.029556588
# Number of sentences: 3761
# Number of words: 200582
# Number of predicted probabilities: 82430
# Cross entropy (base e): 5.071950370897041
# Perplexity: 159.485079261145
# ./01-models.sh  5353.70s user 2027.92s system 99% cpu 2:03:12.38 total
