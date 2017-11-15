#!/bin/bash -e

set -o pipefail

lm_scale=12
prune_beam=8
beam=650
max_tokens_per_node=200
recombination_order=20
shortlist=true
cmd=run.pl

echo "${0} ${@}"  # Print the command line for logging

[ -f ./path.sh ] && source ./path.sh
source utils/parse_options.sh

if [ "${#}" -ne 4 ]
then
   cat <<EOF
Rescores lattices using a TheanoLM model. Kaldi lattices contain "graph
scores" that consist of e.g. silence and pronunciation probabilities in addition
to language model probabilities, so this script first removes the original LM
scores from the graph scores to obtain the remainder of the probability mass.
Then the NNLM scores are interpolated with the graph scores with 0.5 weight, and
LM scale is doubled to have the effect of adding the graph scores to the NNLM
scores. As a consequence, the LM scores are completely replaced, i.e.
interpolation with the original probabilities have to be done afterwards.

Usage:
  steps/lmrescore_theanolm_lattices.sh [options] <lang-dir> <nnlm> \
                                       <input-decode-dir> <output-decode-dir>

Options:
  --N N
      Generate N hypotheses per utterance. (default: 10)

  --lm-scale WEIGHT
      Scale LM scores with WEIGHT when decoding the lattices. (default: 12)

  --prune-beam BEAM
      Prune the lattices with BEAM before rescoring them. (default: 8)

  --beam BEAM
      When decoding the lattices, drop a token if the difference between its
      score and the best score is larger thane BEAM. (default: 650)

  --max-tokens-per-node N
      Limit to N tokens in each lattice node when decoding. (default: 200)

  --recombination-order N
      Keep only the best token if the N previous words traversed by the tokens
      are identical. (default: 20)

  --shortlist (true|false)
      If true, <unk> token probability will be distributed among the
      out-of-shortlist words according to their unigram frequencies in the
      training data. (default: true)

  --cmd COMMAND
      Submit parallel jobs to a cluster using COMMAND, typically run.pl or
      queue.pl. (default: run.pl)
EOF
   exit 1
fi

lang_dir="${1}"
nnlm="${2}"
in_dir="${3}"
out_dir="${4}"

script_name=$(basename "${0}")
lm_scale_x2=$(perl -e "print 2 * ${lm_scale}")
old_lm="${lang_dir}/G.carpa"

for file in "${old_lm}" "${nnlm}" "${in_dir}/lat.1.gz"
do
    if [ ! -f "${file}" ]
    then
        echo "${script_name}: expected file ${file} to exist."
        exit 1
    fi
done

nj=$(cat "${in_dir}/num_jobs")
mkdir -p "${out_dir}"
echo "${nj}" >"${out_dir}/num_jobs"
mkdir -p "${out_dir}/log"

declare -a theanolm_args=()
[ "${shortlist}" = true ] && theanolm_args+=(--shortlist)

${cmd} "JOB=1:${nj}" "${out_dir}/log/lmrescore_theanolm.JOB.log" \
  gunzip -c "${in_dir}/lat.JOB.gz" \| \
  lattice-prune \
    --inv-acoustic-scale="${lm_scale}" \
    --beam="${prune_beam}" \
    ark:- ark:- \| \
  lattice-lmrescore-const-arpa \
    --lm-scale=-1.0 \
    ark:- "${old_lm}" ark,t:- \| \
  theanolm decode ${nnlm} \
    --lattice-format kaldi \
    --kaldi-vocabulary "${lang_dir}/words.txt" \
    --output kaldi \
    --nnlm-weight 0.5 \
    --lm-scale "${lm_scale_x2}" \
    --max-tokens-per-node "${max_tokens_per_node}" \
    --beam "${beam}" \
    --recombination-order "${recombination_order}" \
    "${theanolm_args[@]}" \
    --log-file "${out_dir}/log/theanolm_decode.JOB.log" \
    --log-level debug \| \
  lattice-minimize ark:- ark:- \| \
  gzip -c \>"${out_dir}/lat.JOB.gz"

echo "${script_name}: Finished."
exit 0
