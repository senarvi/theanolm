#!/bin/bash -e

cmd="run.pl"
decode_mbr="false"
min_lm_scale="9"
max_lm_scale="20"
wi_penalties="0.0 0.5 1.0"
beam="6"
glm=""
stage="0"

echo "${0} ${@}"  # Print the command line for logging.

[ -f ./path.sh ] && source ./path.sh
source utils/parse_options.sh

if [ "${#}" -ne 3 ]
then
    cat <<EOF
Decodes lattices and scores using sclite.

Usage:
  steps/score_sclite.sh [options] <lang-dir> <decode-dir> <trn>

Options:
  --cmd COMMAND
      Submit parallel jobs to a cluster using COMMAND, typically run.pl or
      queue.pl. (default: run.pl)

  --decode-mbr (true|false)
      Perform Minimum Bayes Risk (confusion network) decoding. (default: false)

  --min-lm-scale WEIGHT
      Smallest language model scale to try. (default: 9)

  --max-lm-scale WEIGHT
      Largest language model scale to try. (default: 20)

  --wi-penalties WIPS
      Decode using the word insertion penalties given in the whitespace-
      separated list WIPS. (default: "0.0 0.5 1.0")

  --glm GLM
      Applies the set of text transformation rules contained in the 'global map
      file' GLM to the hypotheses. (default: no normalization)

  --stage N
      Continue execution from stage N. (default: 0)
EOF
   exit 1
fi

lang_dir="${1}"
dir="${2}"
ref_trn="${3}"

command -v sclite >/dev/null || { echo "Command sclite was not found."; exit 1; }
command -v csrfilt.sh >/dev/null || { echo "Command csrfilt.sh was not found."; exit 1; }
[ -f "${ref_trn}" ] || { echo "Reference file ${ref_trn} was not found."; exit 1; }

score_dir="${dir}/scoring_sclite"
mkdir -p "${score_dir}"

for wip in ${wi_penalties}
do
    if [ "${decode_mbr}" = true ]
    then
        ${cmd} "LMWT=${min_lm_scale}:${max_lm_scale}" "${score_dir}/penalty_${wip}/log/best_path.LMWT.log" \
          lattice-scale \
            --inv-acoustic-scale=LMWT \
            "ark:gunzip -c ${dir}/lat.*.gz |" \
            ark:- \| \
          lattice-add-penalty --word-ins-penalty="${wip}" ark:- ark:- \| \
          lattice-prune --beam="${beam}" ark:- ark:- \| \
          lattice-mbr-decode \
            --word-symbol-table="${lang_dir}/words.txt" \
            ark:- \
            ark,t:- \| \
          utils/int2sym.pl -f 2- "${lang_dir}/words.txt" \
          ">${score_dir}/penalty_${wip}/LMWT.ref"
    else
        ${cmd} "LMWT=${min_lm_scale}:${max_lm_scale}" "${score_dir}/penalty_${wip}/log/best_path.LMWT.log" \
          lattice-scale \
            --inv-acoustic-scale=LMWT \
            "ark:gunzip -c ${dir}/lat.*.gz |" \
            ark:- \| \
          lattice-add-penalty --word-ins-penalty="${wip}" ark:- ark:- \| \
          lattice-best-path \
            --word-symbol-table="${lang_dir}/words.txt" \
            ark:- \
            ark,t:- \| \
          utils/int2sym.pl -f 2- "${lang_dir}/words.txt" \
          ">${score_dir}/penalty_${wip}/LMWT.ref"
    fi
done


normalize_hyp () {
    if [ -n "${1}" ]
    then
        csrfilt.sh -dh -i trn -t hyp "${1}"
    else
        cat
    fi
}

# sclite requires transcripts in an 8-bit encoding.
ref_trn_iso="${score_dir}/reference.trn.iso"
iconv -f UTF-8 -t ISO-8859-15 <"${ref_trn}" >"${ref_trn_iso}"

for wip in ${wi_penalties}
do
    for lmwt in $(seq ${min_lm_scale} ${max_lm_scale})
    do
        hyp="${score_dir}/penalty_${wip}/${lmwt}.ref"
        hyp_trn_iso="${hyp}.iso"
        awk '{ $(NF+1) = "(" $1 ")"; $1 = ""; } sub(FS, "")' "${hyp}" |
          iconv -f UTF-8 -t ISO-8859-15 |
          normalize_hyp "${glm}" \
          >"${hyp_trn_iso}"

        sclite -r "${ref_trn_iso}" trn -h "${hyp_trn_iso}" trn -i rm -o all -o dtl
    done
done
