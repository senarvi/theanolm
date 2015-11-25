# TheanoLM


## Introduction

TheanoLM is a recurrent neural network language modeling tool implemented using
the Python library [Theano](http://www.deeplearning.net/software/theano/).
Theano allows the user to customize and extend the neural network very
conveniently, still generating highly efficient code that can utilize GPUs or
multiple CPUs for parallel computation. TheanoLM already supports different
network sizes and architectures, including long short-term memory and gated
recurrent units. New gradient based optimization methods can be easily
implemented. Currently Stochastic Gradient Descent, RMSProp, AdaGrad, ADADELTA,
and Adam optimizers are implemented.


## Installation

To run the program, you need to first install Theano. The Python package
theanolm has to be found from a directory on your `$PYTHONPATH`, and the scripts
from bin directory have to be found from a directory on your `$PATH`. The
easiest way to try the program is to clone the Git repository to, say,
`$HOME/git/theanolm`, add that directory to `$PYTHONPATH` and the `bin`
subdirectory to `$PATH`:

    mkdir -p "$HOME/git"
    cd "$HOME/git"
    git clone https://github.com/senarvi/theanolm.git
    export PYTHONPATH="$PYTHONPATH:$HOME/git/theanolm"
    export PATH="$PATH:$HOME/git/theanolm/bin"


### Using a GPU

Theano can automatically utilize NVIDIA GPUs for numeric computation. First you
need to have CUDA installed. A GPU device can be selected using `$THEANO_FLAGS`
environment variable, or in `.theanorc` configuration file. For details about
configuring Theano, see
[Theano manual](http://deeplearning.net/software/theano/library/config.html).
Currently Theano supports only 32-bit floating point precision, when using a
GPU. The simplest way to get started is to set `$THEANO_FLAGS` as follows:

    export THEANO_FLAGS=floatX=float32,device=gpu


## Usage

`theanolm` command recognizes several subcommands:

- `theanolm train` trains a neural network language model.
- `theanolm score` performs text scoring and perplexity computation using a
  neural network language model.
- `theanolm sample` generates sentences by sampling words from a neural network
  language model.
- `theanolm version` displays the version number and exits.

The complete list of command line options available for each subcommand can be
displayed with the `--help` argument, e.g.

    theanolm train --help


### Training a language model

#### Dictionary

A model can be trained using words or word classes. For larger model and data
sizes word classes are generally necessary to keep the computational cost of
training and evaluating models reasonable.

A dictionary is provided to the training script. If words are used, the
dictonary is simply a list of words, one per line, and `--dictionary-format
words` argument is given to `theanolm train` command. Words that do not appear
in the dictionary will be mapped to the <UNK> token.

If you want to use word classes,
[SRILM format](http://www.speech.sri.com/projects/srilm/manpages/classes-format.5.html)
for word class definitions is recommended. Each line in the dictionary contains
a class name, class membership probability, and a word. `--dictionary-format
srilm-classes` argument is given to `theanolm train` command. The program also
accepts simple `classes` format without class membership probabilities, but it
is not currently able to learn the class membership probabilities from the
training data.

#### Optimization

The objective of the implemented optimization methods is to maximize the
likelihood of the training sentences. The cost function is the sum of the
negative log probabilities of the training words, given the preceding input
words.

Training words are processed in sequences that by default correspond to lines of
training data. Maximum sequence length may be given with the `--sequence-length`
argument, which limits the time span for which the network can learn
dependencies.

All the implemented optimization methods are based on Gradient Descent, meaning
that the neural network parameters are updated by taking steps proportional to
the negative of the gradient of the cost function. The true gradient is
approximated by subgradients on subsets of the training data called
“mini-batches”.

The size of the step taken when updating neural network parameters is controlled
by “learning rate”. The initial value can be set using the `--learning-rate`
argument. The average per-word gradient will be multiplied by this factor. In
practice the gradient is scaled by the number of words by dividing the cost
function by the number of training examples in the mini-batch. In most of the
cases, something between 0.001 and 1.0 works well, depending on the optimization
method. For example, Adam seems to require a lower value than the other methods.

The number of sequences included in one mini-batch can be set with the
`--batch-size` argument. Larger mini-batches are more efficient to compute on a
GPU, and result in more reliable gradient estimates. However, when a larger
batch size is selected, the learning rate may have to be reduced to keep the
optimization stable. This makes a too large batch size inefficient. Usually a
value between 4 and 32 is used.

#### Command line

Train command takes four positional arguments: output model path, training data
path, validation data path, and dictionary path. The model will be saved in
NumPy .npz format. The input files can be either plain text or compressed with
gzip. Text data is read one utterance per line. Start-of-sentence and
end-of-sentence tags (`<s>` and `</s>`) will be added to the beginning and end
of each utterance, if they are missing. If an empty line is encountered, it will
be ignored, instead of interpreted as the empty sentence `<s> </s>`.

Below is an example of how to train a language model, assuming you have the word
classes in SRILM format in `dictionary.classes`:

    theanolm train \
      model.npz \
      training-data.txt.gz \
      validation-data.txt.gz \
      dictionary.classes \
      --dictionary-format srilm-classes \
      --hidden-layer-size 300 \
      --hidden-layer-type lstm \
      --optimization-method adam \
      --batch-size 16 \
      --learning-rate 0.01

During training, TheanoLM will save `model.npz` every time a minimum of the
validation set cost is found. The file contains the current values of the model
parameters and the training hyperparameters. If the file exists already when the
training starts, and the saved model is compatible with the specified command
line arguments, TheanoLM will automatically continue training from the previous
state.


### Scoring a text file

Score command takes three positional arguments: input model path, evaluation
data path, and dictionary path. Evaluation data is processed identically to the
training and validation data, i.e. explicit start-of-sentence and
end-of-sentence tags are not necessary in the beginning and end of each
utterance, except when one wants to compute the probability of the empty
sentence `<s> </s>`.

The level of detail can be controlled by the `--output` parameter. The value can
be one of:

- `perplexity` -- Compute perplexity and other statistics of the entire corpus.
- `word-scores` -- Display log probability scores of each word, in addition to
  sentence and corpus perplexities.
- `utterances-scores` -- Write just the log probability score of each utterance,
  one per line. This can be used for rescoring n-best lists.

Below is an example of how to compute the perplexity of a model on evaluation
data:

    theanolm score \
      model.npz \
      evaluation-data.txt.gz \
      dictionary.classes \
      --dictionary-format srilm-classes \
      --output perplexity


### Generating text using a model

A neural network language model can also be used to generate text, using the
`theanolm sample` command:

    theanolm sample \
      model.npz \
      dictionary.classes \
      --dictionary-format srilm-classes
      --num-sentences 10


## About the project

TheanoLM is open source and licensed under the
[Apache License, Version 2.0](LICENSE.txt).


### Contributing

You're welcome to contribute.

1. Fork the repository on GitHub.
2. Clone the forked repository into a local directory: `git clone my-repository-url'
3. Create a new branch: `git checkout -b my-new-feature`
4. Commit your changes: `git commit -a`
5. Push to the branch: `git push origin my-new-feature`
6. Submit a pull request on GitHub.


### Structure of the source code

`theanolm.commands` package contains the main scripts for launching the
subcommands.

`theanolm.network.Network` class stores the neural network architecture and
state. The architecture is built from layers that are implemented in
`theanolm.layers` package. Each layer implements functions for constructing the
symbolic layer structure, which may be different for mini-batch training and
text sampling.

`theanolm.trainers` package contains classes that perform the training
iterations. They are responsible for cross-validation and learning rate
adjustment. They use one of the optimizers found in `theanolm.optimizers`
package to perform the actual parameter update.

`theanolm.textscorer.TextScorer` class is used to score text, both for
cross-validation during training and by the score command for evaluating text.
`theanolm.textsampler.TextSampler` class is used by the sample command for
generating text.


### Author

Seppo Enarvi  
http://users.marjaniemi.com/seppo/
