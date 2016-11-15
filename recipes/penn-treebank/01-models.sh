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
#NUM_NOISE_SAMPLES=3
#LEARNING_RATE=10
#rm -f "${OUTPUT_DIR}/nnlm.h5"
#train
#compute_perplexity

# Model performance stopped improving. Decreasing learning rate from 0.15625 to 0.078125 and resetting state to 100 % of epoch 19.
# Finished training epoch 19. Best validation perplexity 146.85.
# Best validation set perplexity: 146.850085988
# Number of sentences: 3761
# Number of words: 200582
# Number of predicted probabilities: 82430
# Cross entropy (base e): 4.951415101841734
# Perplexity: 141.37488229199442
# ./01-models.sh  11131.72s user 3069.75s system 100% cpu 3:56:41.34 total

#ARCHITECTURE_FILE="${arch_dir}/word-lstm256.arch"
#COST=nce
#NUM_NOISE_SAMPLES=5
#LEARNING_RATE=25
#rm -f "${OUTPUT_DIR}/nnlm.h5"
#train
#compute_perplexity

# Finished training epoch 20. Best validation perplexity 158.46.
# Stopping because 20 epochs was reached.
# Best validation set perplexity: 158.533356181
# Number of sentences: 3761
# Number of words: 200582
# Number of predicted probabilities: 82430
# Cross entropy (base e): 5.018736468023775
# Perplexity: 151.22011158431908
# ./01-models.sh  13962.08s user 3514.67s system 100% cpu 4:51:16.59 total

## noise-contrastive estimation with shared noise samples ######################

#ARCHITECTURE_FILE="${arch_dir}/word-lstm256.arch"
#COST=nce-shared
#NUM_NOISE_SAMPLES=100
#LEARNING_RATE=1
#rm -f "${OUTPUT_DIR}/nnlm.h5"
#train
#compute_perplexity

# Finished training epoch 20. Best validation perplexity 175.32.
# Stopping because 20 epochs was reached.
# Best validation set perplexity: 172.261715319
# Number of sentences: 3761
# Number of words: 200582
# Number of predicted probabilities: 82430
# Cross entropy (base e): 5.089891811854721
# Perplexity: 162.37229434582747
# ./01-models.sh  4278.18s user 1731.88s system 99% cpu 1:40:13.76 total
