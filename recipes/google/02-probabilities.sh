#!/bin/bash -e

script_dir=$(dirname "${0}")
script_dir=$(readlink -e "${script_dir}")

source "${script_dir}/paths.sh"

[ -n "${OUTPUT_DIR}" ] || { echo "OUTPUT_DIR required." >&2; exit 1; }

counts_file="${OUTPUT_DIR}/word-counts"
if [ ! -s "${counts_file}" ]
then
	echo "${counts_file}"
	cat "${TRAIN_FILES[@]}" |
	  ngram-count -order 1 -no-sos -no-eos -text - -write "${counts_file}"
fi

classes_file="${OUTPUT_DIR}/classes"
echo "${classes_file}"
"${script_dir}/../common/fix-class-probabilities.py" \
  <(zcat "${OUTPUT_DIR}/exchange.classes.gz") "${counts_file}" |
  grep -v ' 0.0 ' \
  >"${classes_file}"
