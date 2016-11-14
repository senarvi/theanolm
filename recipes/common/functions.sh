#!/bin/bash -e
#
# Functions for estimating language models.

select_vocabulary () {
	local vocab_file="${1}"

	[ "${#TRAIN_FILES[@]}" -lt 1 ] && { echo "TRAIN_FILES required." >&2; exit 1; }

	command -v ngram-count >/dev/null 2>&1 || { echo >&2 "ngram-count not found. Please install from http://www.speech.sri.com/projects/srilm/."; exit 1; }

	if [ -n "${VOCAB_MIN_COUNT}" ]
	then
		cat "${TRAIN_FILES[@]}" |
		  ngram-count -order 1 -text - -no-sos -no-eos -write - |
		  egrep -v '^(<unk>|<s>|</s>|-pau-)' |
		  sort -s -g -k 2,2 -r |
		  awk '$2 >= '"${VOCAB_MIN_COUNT}"' { print $1 }' \
		  >"${vocab_file}"
	else
		cat "${TRAIN_FILES[@]}" |
		  ngram-count -order 1 -text - -no-sos -no-eos -write-vocab - |
		  egrep -v '^(<unk>|<s>|</s>|-pau-)' |
		  sort \
		  >"${vocab_file}"
	fi
}

create_classes () {
	[ "${#TRAIN_FILES[@]}" -lt 1 ] && { echo "TRAIN_FILES required." >&2; exit 1; }
	[ -n "${OUTPUT_DIR}" ] || { echo "OUTPUT_DIR required." >&2; exit 1; }

	local num_classes="${NUM_CLASSES:-2000}"
	local num_threads="${NUM_THREADS:-4}"
	local max_seconds="${EXCHANGE_MAX_SECONDS:-430000}"  # Default time limit 5 days.

	command -v exchange >/dev/null 2>&1 || { echo >&2 "exchange not found. Please install from https://github.com/aalto-speech/exchange."; exit 1; }

	declare -a extra_args

	mkdir -p "${OUTPUT_DIR}"

	local temp_file=$(ls -1 "${OUTPUT_DIR}"/exchange.temp*.classes.gz 2>/dev/null |
	                  sort -V |
	                  tail -1)
	if [ -s "${temp_file}" ]
	then
		echo "Continuing from ${temp_file}."
		extra_args+=(--class-init="${temp_file}")
	elif [ -n "${READ_CLASSES}" ]
	then
		local classes_init="${OUTPUT_DIR}/classes.init"
		cat <<EOF >"${classes_init}"
0: <s>,</s>
1: <unk>
EOF
		awk '{
		    classes[$2] = classes[$2] $1 ",";
		  }
		  END {
		    new_id = 2;
		    for (old_id in classes)
		      print new_id++ ": " classes[old_id];
		  }' \
		  "${READ_CLASSES}" |
		  sed 's/,$//' \
		  >>"${classes_init}"
		extra_args+=(--class-init="${classes_init}")
	fi

	local train_file="${OUTPUT_DIR}/train.txt"
	echo "${train_file}"
	cat "${TRAIN_FILES[@]}" >"${train_file}"

	local vocab_file="${OUTPUT_DIR}/cluster.vocab"
	echo "${vocab_file}"
	[ -s "${vocab_file}" ] || select_vocabulary "${vocab_file}"

	(set -x; exchange \
	  --num-classes="${num_classes}" \
	  --max-time="${max_seconds}" \
	  --num-threads="${num_threads}" \
	  --vocabulary="${vocab_file}" \
	  "${extra_args[@]}" \
	  "${train_file}" \
	  "${OUTPUT_DIR}/exchange")

	rm -f "${train_file}" "${OUTPUT_DIR}"/exchange.temp*
	echo "create_classes finished."
}

train () {
	[ "${#TRAIN_FILES[@]}" -lt 1 ] && { echo "TRAIN_FILES required." >&2; exit 1; }
	[ -n "${DEVEL_FILE}" ] || { echo "DEVEL_FILE required." >&2; exit 1; }
	[ -n "${OUTPUT_DIR}" ] || { echo "OUTPUT_DIR required." >&2; exit 1; }
	[ -n "${ARCHITECTURE_FILE}" ] || { echo "ARCHITECTURE_FILE required." >&2; exit 1; }

	command -v theanolm >/dev/null 2>&1 || { echo >&2 "theanolm not found. Please install from https://github.com/senarvi/TheanoLM."; exit 1; }

	local sequence_length="${SEQUENCE_LENGTH:-100}"
	local batch_size="${BATCH_SIZE:-16}"
	local training_strategy="${TRAINING_STRATEGY:-local-mean}"
	local optimization_method="${OPTIMIZATION_METHOD:-adagrad}"
	local stopping_criterion="${STOPPING_CRITERION:-annealing-count}"
	local cost="${COST:-cross-entropy}"
	local learning_rate="${LEARNING_RATE:-0.1}"
	local gradient_decay_rate="${GRADIENT_DECAY_RATE:-0.9}"
	local epsilon="${EPSILON:-1e-6}"
        local num_noise_samples="${NUM_NOISE_SAMPLES:-1}"
	local validation_freq="${VALIDATION_FREQ:-8}"
	local patience="${PATIENCE:-4}"
	local run_gpu="${RUN_GPU}"

	declare -a extra_args
	[ -n "${MAX_GRADIENT_NORM}" ] && extra_args+=(--gradient-normalization "${MAX_GRADIENT_NORM}")
	if [ -n "${IGNORE_UNK}" ]
        then
		extra_args+=(--unk-penalty 0)
	elif [ -n "${UNK_PENALTY}" ]
	then
		extra_args+=(--unk-penalty "${UNK_PENALTY}")
	fi
	[ -n "${DEBUG}" ] && extra_args+=(--debug)
	[ -n "${ARCHITECTURE_FILE}" ] && extra_args+=(--architecture "${ARCHITECTURE_FILE}")

	mkdir -p "${OUTPUT_DIR}"

	# Tell Theano to use GPU.
	export THEANO_FLAGS="floatX=float32,device=gpu,nvcc.fastmath=True"

	# Taining vocabulary or classes.
	local vocab_file
	local vocab_format
	if [ -n "${CLASSES}" ]
	then
		vocab_file="${CLASSES}"
		vocab_format="srilm-classes"
	else
		vocab_file="${OUTPUT_DIR}/nnlm.vocab"
		echo "${vocab_file}"
		[ -s "${vocab_file}" ] || select_vocabulary "${vocab_file}"
		vocab_format="words"
	fi

	(set -x; theanolm train \
	  "${OUTPUT_DIR}/nnlm.h5" \
	  "${DEVEL_FILE}" \
	  --training-set "${TRAIN_FILES[@]}" \
	  --vocabulary "${vocab_file}" \
          --vocabulary-format "${vocab_format}" \
	  --sequence-length "${sequence_length}" \
	  --batch-size "${batch_size}" \
	  --training-strategy "${training_strategy}" \
	  --optimization-method "${optimization_method}" \
	  --stopping-criterion "${stopping_criterion}" \
	  --cost "${cost}" \
	  --learning-rate "${learning_rate}" \
	  --gradient-decay-rate "${gradient_decay_rate}" \
	  --numerical-stability-term "${epsilon}" \
	  --num-noise-samples "${num_noise_samples}" \
	  --validation-frequency "${validation_freq}" \
	  --patience "${patience}" \
	  --max-epochs 20 \
	  --min-epochs 1 \
	  --random-seed 1 \
	  --log-level debug \
	  --log-interval 1000 \
          "${extra_args[@]}")
	echo "train finished."
}

compute_perplexity () {
	[ -n "${EVAL_FILE}" ] || { echo "EVAL_FILE required." >&2; exit 1; }
	[ -n "${OUTPUT_DIR}" ] || { echo "OUTPUT_DIR required." >&2; exit 1; }

	declare -a extra_args
	if [ -n "${IGNORE_UNK}" ]
        then
		extra_args+=(--unk-penalty 0)
	elif [ -n "${UNK_PENALTY}" ]
	then
		extra_args+=(--unk-penalty "${UNK_PENALTY}")
	fi

	# Tell Theano to use GPU.
	export THEANO_FLAGS="floatX=float32,device=gpu,nvcc.fastmath=True"

	local vocab_file
	local vocab_format
	if [ -n "${CLASSES}" ]
	then
		vocab_file="${CLASSES}"
		vocab_format="srilm-classes"
	else
		vocab_file="${OUTPUT_DIR}/nnlm.vocab"
		vocab_format="words"
	fi

	(set -x; theanolm score \
	  "${OUTPUT_DIR}/nnlm.h5" \
	  "${EVAL_FILE}" \
	  --output "perplexity" \
	  "${extra_args[@]}")
}
