#!/bin/bash -e
#SBATCH --partition gpu
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=4G

#
# Examples for training TheanoLM models on Penn Treebank corpus. The results (in
# the log files) have been obtained using the processed data that is distributed
# with RNNLM basic examples.
#
# This scripts demonstrates model parallelism. Thean applies various
# optimizations by default that make the scan() function (used to implement
# recurrency) fail with multiple devices. The optimizations have to be disabled
# in order to use multiple GPUs, which makes it slower than using a single GPU.
#

if [ -z "${SLURM_SUBMIT_DIR}" ]
then
	script_dir=$(dirname "${0}")
	script_dir=$(readlink -e "${script_dir}")
else
	script_dir="${SLURM_SUBMIT_DIR}"
fi
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

# Configure Theano to use two GPUs. Optimizations of the scan function will fail
# when using multiple devices.
declare -a DEVICES=(cuda0 cuda1)
source "${script_dir}/../common/configure-theano.sh"
export THEANO_FLAGS="${THEANO_FLAGS},optimizer=None"

# Set training parameters.
OPTIMIZATION_METHOD=adagrad
L2_REGULARIZATION=0.00001
MAX_GRADIENT_NORM=5
STOPPING_CRITERION=no-improvement
VALIDATION_FREQ=1
PATIENCE=0
ARCHITECTURE_FILE="${arch_dir}/word-lstm256-2dev.arch"
COST=cross-entropy
LEARNING_RATE=1
#DEBUG=1

rm -f "${OUTPUT_DIR}/nnlm.h5"
mv -f "${script_dir}/2dev.log" "${script_dir}/2dev.log~" 2>/dev/null || true
train | tee "${script_dir}/2dev.log"
compute_perplexity | tee --append "${script_dir}/2dev.log"
