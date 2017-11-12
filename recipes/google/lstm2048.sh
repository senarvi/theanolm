#!/bin/bash -e
#SBATCH --partition gpu
#SBATCH --time=5-00
#SBATCH --mem=14G
#SBATCH --gres=gpu:teslap100:1

#
# Examples for training TheanoLM models on the 1 Billion Word Language Model
# Benchmark. The vocabulary is limited to words that appear at least three
# times. This recipe uses word classes. For convenience, classes that have been
# created using the exchange algorithm are provided with the repository.
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
OUTPUT_DIR="${OUTPUT_DIR}-lstm2048"

# Load common functions.
source "${script_dir}/../common/functions.sh"
source "${script_dir}/../common/configure-theano.sh"

# Set training parameters.
BATCH_SIZE=256
VOCAB_MIN_COUNT=3
OPTIMIZATION_METHOD=adagrad
MAX_GRADIENT_NORM=5
STOPPING_CRITERION=no-improvement
VALIDATION_FREQ=15
PATIENCE=1
ARCHITECTURE_FILE="${arch_dir}/class-lstm2048.arch"
CLASSES="${OUTPUT_DIR}/classes"
COST=cross-entropy
LEARNING_RATE=1

#mv -f "${script_dir}/classes.log" "${script_dir}/classes.log~" 2>/dev/null || true
#create_classes 2>&1 | tee "${script_dir}/classes.log"
mkdir -p "${OUTPUT_DIR}"
cp "${script_dir}/exchange.classes" "${OUTPUT_DIR}/classes"

rm -f "${OUTPUT_DIR}/nnlm.h5"
mv -f "${script_dir}/lstm2048.log" "${script_dir}/lstm2048.log~" 2>/dev/null || true
train | tee "${script_dir}/lstm2048.log"

compute_perplexity | tee --append "${script_dir}/lstm2048.log"
