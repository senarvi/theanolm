#!/bin/bash -e
#
# Compute language model scores using TheanoLM. Takes a TheanoLM model, location
# for a temporary directory, a text file with the sentences to score, and an
# output file location where to put the scores (negated logprobs) of each
# sentence.
#
# This script uses the Kaldi-style "archive" format, so the input and output
# files will have a first field that corresponds to utterance ID. In N-best
# lists the utterance ID contains a postfix -1, -2, etc. to identify the
# hypotheses.

[ -f ./path.sh ] && source ./path.sh
. utils/parse_options.sh

script_name=$(basename "$0")

if ! command -v theanolm >/dev/null
then
    echo "theanolm was not found"
    exit 1
fi

if [ "${#}" -ne 5 ]
then
    echo "Usage: ${script_name} <nnlm> <temp-dir> <input-text> <output-scores>"
    exit 1
fi

nnlm="${1}"
temp_dir="${2}"
text_in="${3}"
scores_out="${4}"

for x in "${nnlm}" "${text_in}"
do
    if [ ! -f "${x}" ]
    then
        echo "${script_name}: expected file ${x} to exist."
        exit 1
    fi
done

mkdir -p "${temp_dir}"
cut -d' ' -f2- "${text_in}" |
  awk '$1 != "<s>" { print "<s>", $0 }' |
  awk '$NF != "</s>" { print $0, "</s>" }' \
  >"${temp_dir}/text"
cut -d' ' -f1 "${text_in}" >"${temp_dir}/ids"

(set -x; theanolm score \
  "${nnlm}" \
  "${temp_dir}/text" \
  --output-file "${temp_dir}/scores" \
  --output utterance-scores)

num_ids=$(wc -l <"${temp_dir}/ids")
num_scores=$(wc -l <"${temp_dir}/scores")
if [ "${num_ids}" -ne "${num_scores}" ]
then
    echo "${script_name}: Only ${num_scores} of ${num_ids} utterances were scored."
    exit 1
fi

paste "${temp_dir}/ids" "${temp_dir}/scores" |
  awk '{ print $1, -$2; }' >"${scores_out}"
echo "${script_name}: Wrote NNLM scores to ${scores_out}."
