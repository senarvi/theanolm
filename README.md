# TheanoLM

TheanoLM is a recurrent neural network language model implemented using the
Python library Theano. Supports long short-term memory and gated recurrent
units. Includes Stochastic Gradient Descent and Adam optimizers.

To run the program, you need to first install Theano. The Python package
theanolm has to be found from a directory on your $PYTHONPATH, and the scripts
from bin directory have to be found from a directory on your $PATH. You can try
running the program by checking out the repository into $HOME/git/theanolm and
setting the environment variables as below:

    export PYTHONPATH="$PYTHONPATH:$HOME/git/theanolm"
    export PATH="$PATH:$HOME/git/theanolm/bin"

Theano can automatically utilize GPUs in the numeric computations. This is
enabled by setting the THEANO_FLAGS environment variable:

    export THEANO_FLAGS=floatX=float32,device=gpu


## Training

Before training you need to construct a dictionary that lists the words that
can be fed to the network. Other words will be mapped to the <UNK> token. If you
want to use word classes, each line of the dictionary should contain a word and
a class ID.

Below is an example of how you can invoke the train command to train a language
model:

    theanolm train \
      model.npz \
      training-data.txt.gz \
      validation-data.txt.gz \
      dictionary.classes \
      --dictionary-format srilm-classes \
      --hidden-layer-size 300 \
      --hidden-layer-type lstm \
      --batch-size 4 \
      --learning-rate 0.001


## Scoring a text file

After training the model state can be loaded and used to compute score for a
text file using the score command:

    theanolm score \
      model.npz \
      evaluation-data.txt.gz \
      dictionary.classes \
      --dictionary-format srilm-classes


## Generating text using a model

After training the model state can be loaded and used to generate text using
the sample command:

    theanolm sample \
      model.npz \
      dictionary.classes \
      --dictionary-format srilm-classes
      --num-sentences 10
