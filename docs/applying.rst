Applying a language model
=========================

Scoring a text corpus
---------------------

Score command takes three positional arguments: input model path, evaluation
data path, and dictionary path. Evaluation data is processed identically to
training and validation data, i.e. explicit start-of-sentence and
end-of-sentence tags are not needed in the beginning and end of each utterance,
except when one wants to compute the probability of the empty sentence
``<s> </s>``.

The level of detail can be controlled by the ``--output`` parameter. The value
can be one of:

* ``perplexity`` – Compute perplexity and other statistics of the
  entire corpus.
* ``word-scores`` – Display log probability scores of each word, in
  addition to sentence and corpus perplexities.
* ``utterances-scores`` – Write just the log probability score of each
  utterance, one per line. This can be used for rescoring n-best lists.

The example below shows how one can compute the perplexity of a model on
evaluation data::

    theanolm score model.h5 test-data.txt.gz --output perplexity

Generating text
---------------

A neural network language model can also be used to generate text, using the
``theanolm sample`` command::

    theanolm sample model.h5 --num-sentences 10
