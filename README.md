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
- `theanolm score` uses a neural network language model to compute perplexity
  score for a text file, or rescore an n-best list.
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
words, divided by the number of words.

The training data file should contain one sentence per line. Training words are
processed in sequences that by default correspond to sentences. Maximum sequence
length may be given with the `--sequence-length` parameter, which limits the
time span for which the network can learn dependencies.

All the implemented optimization methods are based on Gradient Descent, meaning
that the neural network parameters are updated by taking steps proportional to
the negative of the gradient of the cost function. The true gradient is
approximated by subgradients on subsets of the training data called
“mini-batches”. The number of sequences included in one mini-batch can be set
with the `--batch-size` parameter; usually values between 4 and 32 are used.
Larger mini-batches are more efficient to compute on a GPU and may result in a
smoother convergence.

#### Command line

Below is an example of how you can train a language model, assuming you have
already created the word classes in `dictionary.classes`:

    theanolm train \
      model.npz \
      training-data.txt.gz \
      validation-data.txt.gz \
      dictionary.classes \
      --dictionary-format srilm-classes \
      --hidden-layer-size 300 \
      --hidden-layer-type lstm \
      --batch-size 16 \
      --learning-rate 0.001

During training, TheanoLM will save `model.npz` every time a minimum of the
validation set cost is found. The file contains model parameters and values of
training hyperparameters in numpy .npz format. If the file exists already when
the training starts, TheanoLM will automatically continue training from the
previous state.


### Scoring a text file

After training, the model state can be loaded and used to compute a perplexity
score for a text file, using the `theanolm score` command:

    theanolm score \
      model.npz \
      evaluation-data.txt.gz \
      dictionary.classes \
      --dictionary-format srilm-classes


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
