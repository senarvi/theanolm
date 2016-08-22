Applying a language model
=========================

Scoring a text corpus
---------------------

``theanolm score`` command can be used to compute the perplexity of evaluation
data, or to rescore an n-best list by computing the probability of each
sentence. It takes two positional arguments. These specify the path to the
TheanoLM model and the text to be evaluated. Evaluation data is processed
identically to training and validation data, i.e. explicit start-of-sentence and
end-of-sentence tags are not needed in the beginning and end of each utterance,
except when one wants to compute the probability of the empty sentence
``<s> </s>``.

What the command prints can be controlled by the ``--output`` parameter. The
value can be one of:

perplexity
  Compute perplexity and other statistics of the entire corpus.

word-scores
  Display log probability scores of each word, in addition to sentence and
  corpus perplexities.

utterance-scores
  Write just the log probability score of each utterance, one per line. This can
  be used for rescoring n-best lists.

The easiest way to evaluate a model is to compute the perplexity of the model on
evaluation data, lower perplexity meaning a better match. Note that perplexity
values are meaningful to compare only when the vocabularies are identical. If
you want to compare perplexities with back-off model perplexities computed e.g.
using `SRILM <http://www.speech.sri.com/projects/srilm/>`_, note that SRILM
ignores OOV words when computing the perplexity. You get the same behaviour from
TheanoLM, if you use ``--unk-penalty 0``. TheanoLM includes sentence end tokens
in the perplexity computation, so you should look at the ``ppl`` value from
SRILM output. The example below shows how one can compute the perplexity of a
model on evaluation data, while ignoring OOV words::

    theanolm score model.h5 test-data.txt --output perplexity --unk-penalty 0

Probabilities of individual words can be useful for debugging problems. The
``word-scores`` output can be compared to the ``-ppl -debug 2`` output of SRILM.
While the base chosen to represent log probabilities does not affect perplexity,
when comparing log probabilities, the same base has to be chosen. Internally
TheanoLM uses the natural logarithm, and by default also prints the log
probabilities in the natural base. SRILM prints base 10 log probabilities, so in
order to get comparable log probabilities, you should use ``--log-base 10`` with
TheanoLM. The example below shows how one can display individual word scores in
base 10::

    theanolm score model.h5 test-data.txt --output word-scores --log-base 10

Rescoring n-best lists
----------------------

A typical use of a neural network language model is to rescore n-best lists
generated during the first recognition pass. Often a word lattice that
represents the search space can be created as a by-product in an ASR decoder. An
n-best list can be decoded from a word lattice using lattice-tool from SRILM.
Normally there are many utterances, so the lattice files are listed in, say,
``lattices.txt``. The example below reads the lattices in HTK SLF format and
writes 100-best lists to the ``nbest`` directory::

    mkdir nbest
    lattice-tool -in-lattice-list lattices.txt -read-htk -nbest-decode 100 \
                 -out-nbest-dir nbest

It would be inefficient to call TheanoLM on each n-best list separately. A
better approach is to concatenate them into a single file and prefix each line
with the utterance ID::

    for gz_file in nbest/*.gz
    do
        utterance_id=$(basename "${gz_file}" .gz)
        zcat "${gz_file}" | sed "s/^/${utterance_id} /"
    done >nbest-all.txt

lattice-tool output includes the acoustic and language model scores. TheanoLM
needs only the sentences. You should use ``--log-base 10`` if you're rescoring
an n-best list generated using SRILM::

    cut -d' ' -f5- <nbest-all.txt >sentences.txt
    theanolm score model.h5 sentences.txt \
        --output-file scores.txt --output utterance-scores \
        --log-base 10

The resulting file ``scores.txt`` contains one log probability on each line.
These can be simply inserted into the original n-best list, or interpolated with
the original language model scores using some weight *lambda*::

    paste -d' ' scores.txt nbest-all.txt |
    awk -v "lambda=0.5" \
        '{ nnscore = $1; boscore = $4;
           $1 = ""; $4 = nnscore*lambda + boscore*(1-lambda);
           print }' |
    awk '{ $1=$1; print }' >nbest-interpolated.txt

The total score of a sentence can be computed by weighting the language model
scores with some value *lmscale* and adding the acoustic score. The best
sentences from each utterance are obtained by sorting by utterance ID and score,
and taking the first sentence of each utterance. The fields we have in the
n-best file are utterance ID, acoustic score, language model score, and number
of words::

    awk -v "lmscale=14.0" \
        '{ $2 = $2 + $3*lmscale; $3 = $4 = "";
           print }' <nbest-interpolated.txt |
    sort -k1,1 -k2,2gr |
    awk '$1 != id { id = $1; $2 = ""; print }' |
    awk '{ $1=$1; print }' >1best.ref

Decoding word lattices
----------------------

``theanolm decode`` command can be used to decode the best paths directly from
word lattices using neural network language model probabilities. This is more
efficient than creating an intermediate n-best list and rescoring every
sentence::

    theanolm decode model.h5 \
        --lattice-list lattices.txt \
        --output-file 1best.ref --output ref \
        --nnlm-weight 0.5 --lm-scale 14.0

In principle, the context length is not limited in recurrent neural networks, so
an exhaustive search of word lattices would be too expensive. There are three
parameters that constrain the search space by pruning unlikely tokens (partial
hypotheses). These are:

--max-tokens-per-node : N
  Retain at most N tokens at each node. Limiting the number of tokens is very
  effective in cutting the computational cost. Higher values mean higher
  probability of finding the best path, but also higher computational cost. A
  good starting point is 64.

--beam : logprob
  Specifies the maximum log probability difference to the best token at a given
  time. Beam pruning starts to have effect when the beam is smaller than 1000,
  but the effect on word error rate is small before the beam is smaller than
  500.

--recombination-order : N
  When two tokens have identical history up to N previous words, keep only the
  best token. This means effectively that we assume that the influence of a word
  is limited to the probability of the next N words. Recombination seems to have
  little effect on word error rate before N is closer to 20.

The work can be divided to several jobs for a compute cluster, each processing
the same number of lattices. For example, the following SLURM job script would
create an array of 50 jobs. Each would run its own TheanoLM process and decode
its own set of lattices, limiting the number of tokens at each node to 10::

    #!/bin/sh
    #SBATCH --gres=gpu:1
    #SBATCH --array=0-49

    srun --gres=gpu:1 theanolm decode model.h5 \
        --lattice-list lattices.txt \
        --output-file "${SLURM_ARRAY_TASK_ID}.ref" --output ref \
        --nnlm-weight 0.5 --lm-scale 14.0 \
        --max-tokens-per-node 64 --beam 500 --recombination-order 20 \
        --num-jobs 50 --job "${SLURM_ARRAY_TASK_ID}"

Generating text
---------------

A neural network language model can also be used to generate text, using the
``theanolm sample`` command::

    theanolm sample model.h5 --num-sentences 10
