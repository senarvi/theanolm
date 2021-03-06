#!/bin/bash -e
#SBATCH --partition gpu
#SBATCH --time=1-00
#SBATCH --gres=gpu:1
#SBATCH --mem=4G

#
# Examples for training TheanoLM models on Penn Treebank corpus. The results (in
# the log files) have been obtained using the processed data that is distributed
# with RNNLM basic examples. The vocabulary is 10002 words including the <s> and
# </s> symbols. With such a small vocabulary, noise-contrastive estimation does
# not improve training speed. Hierarchical softmax improves training speed with
# only a small degradation in model performance.
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
source "${script_dir}/../common/configure-theano.sh"

# Set training parameters.
OPTIMIZATION_METHOD=adagrad
L2_REGULARIZATION=0.00001
MAX_GRADIENT_NORM=5
STOPPING_CRITERION=no-improvement
VALIDATION_FREQ=1
PATIENCE=0
ARCHITECTURE_FILE="${arch_dir}/word-gcnn-4b.arch"
COST=cross-entropy
LEARNING_RATE=1
#DEBUG=1

rm -f "${OUTPUT_DIR}/nnlm.h5"
mv -f "${script_dir}/gcnn.log" "${script_dir}/gcnn.log~" 2>/dev/null || true
train | tee "${script_dir}/gcnn.log"
compute_perplexity | tee --append "${script_dir}/gcnn.log"
