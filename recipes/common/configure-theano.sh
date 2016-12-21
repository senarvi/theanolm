#!/bin/bash -e
#
# Tell Theano to use as the GPUs specified by $DEVICES (or cuda0 by default).

declare -a devices=("${DEVICES[@]:-cuda0}")
declare -a contexts
for i in "${!devices[@]}"
do
	contexts+=("dev${i}->${devices[${i}]}")
done
THEANO_FLAGS="floatX=float32,device=${devices[0]}"
if [ ${#devices[@]} -gt 1 ]
then
	THEANO_FLAGS=$(IFS=,; echo "${THEANO_FLAGS},contexts=${contexts[*]}")
fi
THEANO_FLAGS="${THEANO_FLAGS},nvcc.fastmath=True"
