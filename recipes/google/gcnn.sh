#!/bin/bash -e
#SBATCH --partition gpu
#SBATCH --time=5-00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G

#
# Examples for training TheanoLM models on the 1 Billion Word Language Model
# Benchmark. The vocabulary is limited to words that appear at least three times
# and <unk> is used to denote the other words.
#

if [ -z "${SLURM_SUBMIT_DIR}" ]
then
	script_dir=$(dirname "${0}")
	script_dir=$(readlink -e "${script_dir}")
else
	script_dir="${SLURM_SUBMIT_DIR}"
fi
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
source "${script_dir}/../common/configure-theano.sh"

# Set training parameters.
BATCH_SIZE=32
VOCAB_MIN_COUNT=3
OPTIMIZATION_METHOD=adagrad
MAX_GRADIENT_NORM=5
STOPPING_CRITERION=no-improvement
VALIDATION_FREQ=25
PATIENCE=4
ARCHITECTURE_FILE="${arch_dir}/word-gcnn-8b-fast.arch"
COST=cross-entropy
LEARNING_RATE=0.8

rm -f "${OUTPUT_DIR}/nnlm.h5"
mv -f "${script_dir}/gcnn.log" "${script_dir}/gcnn.log~" 2>/dev/null || true
train | tee "${script_dir}/gcnn.log"
compute_perplexity | tee --append "${script_dir}/gcnn.log"
