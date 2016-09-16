#!/bin/bash -e

N=10
lm_scale=12
nnlm_weights="0.5 1.0"
cmd=run.pl
use_phi=false
stage=0

echo "${0} ${@}"  # Print the command line for logging.

[ -f ./path.sh ] && source ./path.sh
source utils/parse_options.sh

if [ "${#}" -ne 5 ]
then
   cat <<EOF
Rescores n-best lists using a TheanoLM model. Kaldi lattices contain "graph
scores" that consist of e.g. silence and pronunciation probabilities in addition
to language model probabilities, so this script first removes the original LM
scores from the graph scores to obtain the remainder of the probability mass.

Usage:
  steps/lmrescore_theanolm_nbest.sh [options] <lang-dir> <nnlm> \
                                    <input-decode-dir> <output-decode-dir>

Options:
  --N N
      Generate N hypotheses per utterance. (default: 10)

  --lm-scale WEIGHT
      Scale AM scores with the inverse of WEIGHT in N-best list generation.
      Equivalent to LM scaling. Note: The actual scoring can be done at
      different LM scales, but this should be as close to the optimal value as
      possible to obtain good n-best lists. (default: 12)

  --nnlm-weights WEIGHTS
      Interpolate NNLM scores with each of the weights in the whitespace-
      separated list WEIGHTS. (default: "0.5 1.0")

  --cmd COMMAND
      Submit parallel jobs to a cluster using COMMAND, typically run.pl or
      queue.pl. (default: run.pl)

  --use-phi (true|false)
      Should be set to true, if the source lattices were created by
      lmrescore.sh, and false if they came from decoding. Note: This is kind of
      an obscure option. If true, the script will remove the original LM scores
      using a phi (failure) matcher, which is appropriate if the old LM scores
      were added in this way. Otherwise normal composition is used. This won't
      actually make much difference (if any) to WER, it's more so we know we are
      doing the right thing. (default: false)

  --stage N
      Continue execution from stage N. (default: 0)
EOF
   exit 1
fi

lang_dir="${1}"
nnlm="${2}"
in_dir="${3}"
out_dir="${4}"

script_name=$(basename "${0}")
int2sym="utils/int2sym.pl"
compute_scores="utils/theanolm_compute_scores.sh"
ac_scale=$(perl -e "print 1 / ${lm_scale}")

# Figures out if the old LM is G.fst or G.carpa
if [ -f "${lang_dir}/G.carpa" ]
then
    old_lm="${lang_dir}/G.carpa"
elif [ -f "${lang_dir}/G.fst" ]
then
    old_lm="${lang_dir}/G.fst"
else
    echo "${script_name}: expecting either ${lang_dir}/G.fst or ${lang_dir}/G.carpa to exist"
    exit 1
fi

for file in "${nnlm}" "${in_dir}/lat.1.gz" "${int2sym}" "${compute_scores}"
do
    if [ ! -f "${file}" ]
    then
        echo "${script_name}: expected file ${file} to exist."
        exit 1
    fi
done

nj=$(cat "${in_dir}/num_jobs")
archives_dir="${out_dir}/text_archives"

mkdir -p "${out_dir}"
phi=$(grep -w '#0' "${lang_dir}/words.txt" | awk '{ print $2 }')

rm -f "${out_dir}/.error"
mkdir -p "${out_dir}/log"

if [ "${stage}" -le 0 ]
then
    echo "${script_name}: Converting lattices to N-best lists."

    # Note: the lattice-rmali part here is just because we don't
    # need the alignments for what we're doing.
    $cmd "JOB=1:${nj}" "${out_dir}/log/lattices_to_nbest.JOB.log" \
      lattice-to-nbest \
        --acoustic-scale="${ac_scale}" \
        --n="${N}" \
        "ark:gunzip -c ${in_dir}/lat.JOB.gz |" \
        ark:- \|  \
      lattice-rmali \
        ark:- \
        "ark:|gzip -c >${out_dir}/nbest_with_lmprobs.JOB.gz"
fi

if [ "${stage}" -le 1 ]
then
    echo "${script_name}: Removing original language model probabilities."

    if [ "${old_lm}" = "${lang_dir}/G.fst" ]
    then
        if [ "${use_phi}" = true ]
        then
            # Use the phi-matcher style of composition.. this is appropriate
            # if the old LM scores were added e.g. by lmrescore.sh, using 
            # phi-matcher composition.
            ${cmd} "JOB=1:${nj}" "${out_dir}/log/remove_lmprobs.JOB.log" \
              lattice-scale \
                --acoustic-scale=-1 \
                --lm-scale=-1 \
                "ark:gunzip -c ${out_dir}/nbest_with_lmprobs.JOB.gz |" \
                ark:- \| \
              lattice-compose \
                --phi-label="${phi}" \
                ark:- \
                "${old_lm}" \
                ark:- \| \
              lattice-scale \
                --acoustic-scale=-1 \
                --lm-scale=-1 \
                ark:- \
                "ark:| gzip -c >${out_dir}/nbest_without_lmprobs.JOB.gz"
        else
            # this approach chooses the best path through the old LM FST, while
            # subtracting the old scores.  If the lattices came straight from decoding,
            # this is what we want.  Note here: each FST in "nbest_with_lmprobs.JOB.gz" is a linear FST,
            # it has no alternatives (the N-best format works by having multiple keys
            # for each utterance).  When we do "lattice-1best" we are selecting the best
            # path through the LM, there are no alternatives to consider within the
            # original lattice.
            ${cmd} "JOB=1:${nj}" "${out_dir}/log/remove_lmprobs.JOB.log" \
              lattice-scale \
                --acoustic-scale=-1 \
                --lm-scale=-1 \
                "ark:gunzip -c ${out_dir}/nbest_with_lmprobs.JOB.gz |" \
                ark:- \| \
              lattice-compose \
                ark:- \
                "fstproject --project_output=true ${old_lm} |" \
                ark:- \| \
              lattice-1best \
                ark:- \
                ark:- \| \
              lattice-scale \
                --acoustic-scale=-1 \
                --lm-scale=-1 \
                ark:- \
                "ark:| gzip -c >${out_dir}/nbest_without_lmprobs.JOB.gz"
        fi
    else
        ${cmd} "JOB=1:${nj}" "${out_dir}/log/remove_lmprobs.JOB.log" \
          lattice-lmrescore-const-arpa \
            --lm-scale=-1.0 \
            "ark:gunzip -c ${out_dir}/nbest_with_lmprobs.JOB.gz |" \
            "${old_lm}" \
            "ark:| gzip -c >${out_dir}/nbest_without_lmprobs.JOB.gz"
    fi
fi

if [ "${stage}" -le 2 ]
then
    echo "${script_name}: Extracting text data from the N-best lists without LM probabilities."

    ${cmd} "JOB=1:${nj}" "${out_dir}/log/text_without_lmprobs.JOB.log" \
      mkdir -p "${archives_dir}/JOB" '&&' \
      nbest-to-linear \
        "ark:gunzip -c ${out_dir}/nbest_without_lmprobs.JOB.gz |" \
        "ark,t:${archives_dir}/JOB/ali" \
        "ark,t:${archives_dir}/JOB/words" \
        "ark,t:${archives_dir}/JOB/scores.remainder" \
        "ark,t:${archives_dir}/JOB/scores.am"
fi

if [ "${stage}" -le 3 ]
then
    echo "${script_name}: Extracting text data from the N-best lists with original LM probabilities."
    # This is for interpolation with the original probabilities.

    ${cmd} "JOB=1:${nj}" "${out_dir}/log/text_with_lmprobs.JOB.log" \
      nbest-to-linear \
        "ark:gunzip -c ${out_dir}/nbest_with_lmprobs.JOB.gz |" \
        "ark:/dev/null" \
        "ark:/dev/null" \
        "ark,t:${archives_dir}/JOB/scores.graph" \
        "ark:/dev/null"
fi

if [ "${stage}" -le 4 ]
then
    echo "${script_name}: Creating archives with text-form of words, and LM scores without graph scores."

    for n in $(seq "${nj}")
    do
        "${int2sym}" -f 2- "${lang_dir}/words.txt" \
          <"${archives_dir}/${n}/words" \
	  >"${archives_dir}/${n}/words_text"
        mkdir -p "${archives_dir}/${n}/temp"
        paste "${archives_dir}/${n}/scores.remainder" "${archives_dir}/${n}/scores.graph" |
          awk '{ print $1, ($4 - $2); }' \
          >"${archives_dir}/${n}/scores.oldlm"
    done
fi

if [ "${stage}" -le 5 ]
then
    echo "${script_name}: Invoking TheanoLM to score the sentences."

    ${cmd} "JOB=1:${nj}" "${out_dir}/log/theanolm_compute_scores.JOB.log" \
      "${compute_scores}" \
        "${nnlm}" \
        "${archives_dir}/JOB/temp" \
        "${archives_dir}/JOB/words_text" \
        "${archives_dir}/JOB/scores.nnlm"
fi

if [ "${stage}" -le 6 ]
then
    for nnlm_weight in ${nnlm_weights}
    do
        echo "${script_name}: Interpolating NNLM and original LM scores with NNLM weight ${nnlm_weight}."
        for n in $(seq "${nj}")
        do
            paste "${archives_dir}/${n}/scores.remainder" \
                  "${archives_dir}/${n}/scores.oldlm" \
                  "${archives_dir}/${n}/scores.nnlm" |
              awk -v nnlmweight=${nnlm_weight} \
                '{ key=$1;
                   remainder=$2;
                   lmscore=$4;
                   nnlmscore=$6;
                   score = remainder + (nnlmweight * nnlmscore) + ((1 - nnlmweight) * lmscore);
                   print key, score;
                 } ' >"${archives_dir}/${n}/scores.graph.lambda=${nnlm_weight}"
        done
    done
fi

if [ "${stage}" -le 7 ]
then
    echo "${script_name}: Reconstructing archives back into lattices."

    for nnlm_weight in ${nnlm_weights}
    do
        echo "${out_dir}/nnlm_weight_${nnlm_weight}"
        mkdir -p "${out_dir}/nnlm_weight_${nnlm_weight}"
        ${cmd} "JOB=1:${nj}" "${out_dir}/log/reconstruct_lattice.JOB.log" \
          linear-to-nbest \
            "ark:${archives_dir}/JOB/ali" \
            "ark:${archives_dir}/JOB/words" \
            "ark:${archives_dir}/JOB/scores.graph.lambda=${nnlm_weight}" \
            "ark:${archives_dir}/JOB/scores.am" ark:- \| \
          nbest-to-lattice ark:- "ark:| gzip -c >${out_dir}/nnlm_weight_${nnlm_weight}/lat.JOB.gz"
    done
fi

echo "${script_name}: Finished."
exit 0
