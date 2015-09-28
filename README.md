# TheanoLM

TheanoLM is a recurrent neural network language model implemented using the
Python library Theano. Supports long short-term memory and gated recurrent
units. Includes Stochastic Gradient Descent and Adam optimizers.

Theano can automatically utilize GPUs in the numeric computations. This is
enabled by setting the THEANO_FLAGS environment variable:

    export THEANO_FLAGS=floatX=float32,device=gpu


## Training

Before training you need to construct a dictionary that lists the words that
can be fed to the network. Other words will be mapped to the <UNK> token. If you
want to use word classes, each line of the dictionary should contain a word and
a class ID.

Below is an example of how you can invoke theanolm-train.py to train a language
model:

    theanolm-train.py \
      model.npz \
      training-data.txt.gz \
      validation-data.txt.gz \
      dictionary.classes \
      --dictionary-format srilm-classes \
      --training-state training-state.npz \
      --hidden-layer-size 300 \
      --hidden-layer-type lstm \
      --batch-size 4 \
      --learning-rate 0.001


## Scoring a text file

After training the model state can be loaded and used to compute score for a
text file using theanolm-score.py:

    theanolm-score.py \
      model.npz \
      evaluation-data.txt.gz \
      dictionary.classes \
      --dictionary-format srilm-classes
