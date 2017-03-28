Calling TheanoLM from Python
============================

Scoring an utterance
--------------------

You can also call TheanoLM from a Python script to score utterances. Assuming
you have trained a neural network and saved it in ``model.h5``, first load the
model using ``Network.from_file``::

    from theanolm import Network, TextScorer
    model = Network.from_file('model.h5')

Then create a text scorer. The constructor takes optional arguments concerning
unknown word handling. You might want to ignore unknown words. In that case,
use::

    scorer = TextScorer(model, ignore_unk=True)

Now you can score the text string ``utterance`` using::

    score = scorer.score_line(utterance, model.vocabulary)

Start and end of sentence tags (<s> and </s>) will be automatically inserted to
the beginning and end of the utterance, if they're missing. If the utterance is
empty, None will be returned. Otherwise the returned value is the log
probability of the utterance.
