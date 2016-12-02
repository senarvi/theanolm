Training a language model
=========================

Vocabulary
----------

Because of the softmax normalization performed over the vocabulary at the output
layer of a neural network, vocabulary size has a huge impact on training speed.
Vocabulary size can be reduced by clustering words into classes, and estimating
a language model over the word classes, or using subword units. Another option
is to approximate the softmax normalization using hierarchical softmax,
noise-contrastive estimation, or BlackOut. These options are explained below:

* Class-based models are probably the fastest to train and evaluate, because the
  vocabulary size is usually a few thousand. TheanoLM will use unigram
  probabilities for words inside the classes. TheanoLM is not able to generate
  word classes automatically. You can use for example Percy Liang's
  `brown-cluster`_, *ngram-class* from `SRILM`_, *mkcls* from `GIZA++`_, or
  `word2vec`_ (with *-classes* switch). Creating the word classes can take a
  considerable amount of time.
* A feasible alternative with agglutinative languages is to segment words into
  subword units. For example, a typical vocabulary created with `Morfessor`_ is
  of the order of 10,000 statistical morphs. The vocabulary and training text
  then contain morphs instead of words, and *<w>* token is used to separate
  words.
* A vocabulary as large as hundreds of thousands of words is possible, when
  using hierarchical softmax (*hsoftmax*) output. The output layer is factorized
  into two levels, both performing normalization over an equal number of
  choices. Training will be considerably faster than with regular softmax, but
  the number of parameters will still be large, meaning that the amount of GPU
  memory may limit the usable vocabulary size.
* A new alternative to hierarchical softmax is to approximate softmax by
  sampling a subset of the vocabulary for each mini-batch and contrast the
  correct target words to these *noise* words only, instead of the whole
  vocabulary. Only normal softmax output layer supports sampling. This is
  explained in the `Cost function` section below.

A vocabulary has to be provided for ``theanolm train`` command using the
``--vocabulary`` argument. If classes are not used, the vocabulary is simply a
list of words, one per line, and ``--vocabulary-format words`` argument should
be given. Words that do not appear in the vocabulary will be mapped to the
*<unk>* token. The vocabulary file can also contain classes in one of two
formats, specified by the ``--vocabulary-format`` argument:

* ``classes``  Each line contains a word and an integer class ID. Class
  membership probabilities ``p(word | class)`` are computed as unigram maximum
  likelihood estimates from the training data.
* ``srilm-classes``  Vocabulary file is expected to contain word class
  definitions in `SRILM format
  <http://www.speech.sri.com/projects/srilm/manpages/classes-format.5.html>`_.
  Each line contains a class name, class membership probability, and a word. 

.. _brown-cluster: https://github.com/percyliang/brown-cluster
.. _SRILM: http://www.speech.sri.com/projects/srilm/
.. _GIZA++: https://github.com/moses-smt/giza-pp
.. _word2vec: https://github.com/dav/word2vec
.. _Morfessor: http://morfessor.readthedocs.io/en/latest/

Network structure description
-----------------------------

The neural network layers are specified in a text file. The file contains input
layer elements, one element on each line. Input elements start with the word
*input* and should contain the following fields:

* ``type`` is either *word* or *class* and selects the input unit.
* ``name`` is used to identify the input.

Layer elements start with the word *layer* and may contain the following
fields:

* ``type`` selects the layer class. Has to be specified for all layers.
  Currently *projection*, *tanh*, *lstm*, *gru*, *highwaytanh* (highway network
  layer), *dropout*, *softmax*, and *hsoftmax* (two-level hierarchical softmax)
  are implemented.
* ``name`` is used to identify the layer. Has to be specified for all layers.
* ``input`` specifies a network input or a layer whose output will be the input
  of this layer. Some layers types allow multiple inputs.
* ``size`` gives the number of output connections. If not given, defaults to the
  number of input connections. Will be automatically set to the size of the
  vocabulary in the output layer.
* ``dropout_rate`` may be set in the dropout layer.

The elements have to specified in the order that the network is constructed,
i.e. an element can have in its inputs only elements that have already been
specified. Multiple layers may have the same element in their input. The first
layer should be a projection layer. The last layer is where the network output
will be read from. Description of a typical LSTM neural network language model
could look like this::

    input type=class name=class_input
    layer type=projection name=projection_layer input=class_input size=100
    layer type=lstm name=hidden_layer input=projection_layer size=300
    layer type=softmax name=output_layer input=hidden_layer

A dropout layer is not a real layer in the sense that it does not contain any
neurons. It can be added after another layer, and only sets some activations
randomly to zero at train time. This is helpful with larger networks to prevent
overlearning. The effect can be controlled using the *dropout_rate* parameter.
The training converges slower the larger the dropout rate.

A larger network with dropout layers, word input, and hierarchical softmax
output, could be specified using the following description::

    input type=word name=word_input
    layer type=projection name=projection_layer input=word_input size=500
    layer type=dropout name=dropout_layer_1 input=projection_layer dropout_rate=0.2
    layer type=lstm name=hidden_layer_1 input=dropout_layer_1 size=1500
    layer type=dropout name=dropout_layer_2 input=hidden_layer_1 dropout_rate=0.2
    layer type=tanh name=hidden_layer_2 input=dropout_layer_2 size=1500
    layer type=dropout name=dropout_layer_3 input=hidden_layer_2 dropout_rate=0.2
    layer type=hsoftmax name=output_layer input=dropout_layer_3

Optimization
------------

The objective of the implemented optimization methods is to maximize the
likelihood of the training sentences. All the implemented optimization methods
are based on Gradient Descent, meaning that the neural network parameters are
updated by taking steps proportional to the negative of the gradient of the cost
function. The true gradient is approximated by subgradients on subsets of the
training data called “mini-batches”.

The size of the step taken when updating neural network parameters is controlled
by “learning rate”. The initial value can be set using the ``--learning-rate``
argument. The average per-word gradient will be multiplied by this factor. In
practice the gradient is scaled by the number of words by dividing the cost
function by the number of training examples in the mini-batch. In most of the
cases, something between 0.1 and 1.0 works well, depending on the optimization
method and data.

The number of sequences included in one mini-batch can be set with the
``--batch-size`` argument. Larger mini-batches are more efficient to compute on
a GPU, and result in more reliable gradient estimates. However, when a larger
batch size is selected, the learning rate may have to be reduced to keep the
optimization stable. This makes a too large batch size inefficient. Usually
something like 16 or 32 works well.

Maximum sequence length may be given with the ``--sequence-length`` argument,
which limits the time span for which the network can learn dependencies. Longer
sentences will be split to multiple sequences. If the argument is not given, the
sequences in a mini-batch correspond to sentences. There's no point in using a
value greater than 100, and smaller values such as 25 or 50 can be used to limit
the memory consumption and make the computation more efficient.

The optimization method can be selected using the ``--optimization-method``
argument. Methods that adapt the gradients before updating parameters can
considerably improve the speed of convergence, but training may be less stable.
In order to avoid the gradients exploding, gradient normalization is
recommended. With the ``--max-gradient-norm`` argument one can set the maximum
for the norm of the (adapted) gradients. Typically 5 or 15 works well. The table
below suggests some values for learning rate. Those are a good starting point,
assuming gradient normalization is used.

+--------------------------------+-----------------------+-----------------+
| Optimization Method            | --optimization-method | --learning-rate |
+================================+=======================+=================+
| Stochastic Gradient Descent    | sgd                   | 1               |
+--------------------------------+-----------------------+-----------------+
| Nesterov Momentum              | nesterov              | 1 or 0.1        |
+--------------------------------+-----------------------+-----------------+
| AdaGrad                        | adagrad               | 1 or 0.1        |
+--------------------------------+-----------------------+-----------------+
| ADADELTA                       | adadelta              | 10 or 1         |
+--------------------------------+-----------------------+-----------------+
| SGD with RMSProp               | rmsprop-sgd           | 0.1             |
+--------------------------------+-----------------------+-----------------+
| Nesterov Momentum with RMSProp | rmsprop-nesterov      | 0.01            |
+--------------------------------+-----------------------+-----------------+
| Adam                           | adam                  | 0.01            |
+--------------------------------+-----------------------+-----------------+

AdaGrad automatically scales the gradients before updating the neural network
parameters. It seems to be the fastest method to converge and usually reaches
close to the optimum without manual annealing. ADADELTA is an extension of
AdaGrad that is not as aggressive in scaling down the gradients. It seems to
benefit from manual annealing, but still stay behind AdaGrad in terms of final
model performance. Nesterov Momentum requires manual annealing, but may find a
better final model.

Cost function
-------------

The objective of the optimization can be change by selecting a different cost
function using the ``--cost`` argument. The standard *cross-entropy* cost
involves normalization by computing all the output probabilities. Recently
proposed alternatives, noise-contrastive estimation (*nce*) and BlackOut
(*blackout*), perform normalization only on a subset of the vocabulary during
training. This subset, called noise words, is randomly sampled.

The sampling based costs can be faster to compute, but less stable and slower to
converge. For each data word *k* noise words are sampled, where *k* can be set
using the ``--num-noise-samples`` argument. The higher the number of noise
samples, the more stable and slower the training is.

Creating a different noise sample for every data word is very slow. The noise
sample can be shared across the mini-batch using the ``--noise-sharing``
argument. The value *batch* creates just one noise sample for the entire
mini-batch. The value *seq* creates one noise sample for each time step (word
inside a sequence), but shares the noise samples between sequences. Because of
how multinomial sampling is currently implemented in Theano, noise sharing is
practically necessary and it limits the total number of noise samples per
mini-batch to the vocabulary size.

The distribution where the noise samples are drawn from plays an important role.
Uniform sampling is very fast, but rarely gives good results. It can be selected
by setting the ``--noise-dampening`` argument to zero. Setting that argument to
one corresponds to sampling from the unigram distribution in the training data.
The problem with the unigram distribution is that very rare words may never get
sampled. Usually the optimum value is a bit lower than one.

Command line
------------

Train command takes two mandatory arguments: the output model path and the
``--training-set`` argument followed by path to one or more training data files.
The rest of the arguments have default values. You probably want to provide a
validation text to monitor the progress of the training. Below is an example
that shows what the command line may look like at its simplest::

    theanolm train model.h5 \
      --training-set training-data.txt \
      --validation-file validation-data.txt

The input files can be either plain text or compressed with gzip. Text data is
read one utterance per line. Start-of-sentence and end-of-sentence tags (*<s>*
and *</s>*) will be added to the beginning and end of each utterance, if they
are missing. If an empty line is encountered, it will be ignored, instead of
interpreted as the empty sentence ``<s> </s>``.

The default *lstm300* network architecture is used unless another architecture
is selected with the ``--architecture`` argument. A larger network can be
selected with *lstm1500*, or a path to a custom network architecture description
can be given.

The *no-improvement* stopping condition can be used when validation data is
provided. It halves the learning rate when validation set perplexity stops
improving, and stops training when the perplexity did not improve at all with
the current learning rate. ``--validation-frequency`` argument defines how many
cross-validations are performed on each epoch. ``--patience`` argument defines
how many times perplexity is allowedto increase before learning rate is reduced.

Below is a more complex example that reads word classes from
*vocabulary.classes* and uses Nesterov Momentum optimizer with annealing::

    theanolm train \
      model.h5 \
      --training-set training-data.txt.gz \
      --validation-file validation-data.txt.gz \
      --vocabulary vocabulary.classes \
      --vocabulary-format srilm-classes \
      --architecture lstm1500 \
      --learning-rate 1.0 \
      --optimization-method nesterov \
      --stopping-condition no-improvement \
      --validation-frequency 8 \
      --patience 4

Model file
----------

The model will be saved in HDF5 format. During training, TheanoLM will save the
model every time a minimum of the validation set cost is found. The file
contains the current values of the model parameters and the training
hyperparameters. The model can be inspected with command-line tools such as
h5dump (hdf5-tools Ubuntu package), and loaded into mathematical computation
environments such as MATLAB, Mathematica, and GNU Octave.

If the file exists already when the training starts, and the saved model is
compatible with the specified command line arguments, TheanoLM will
automatically continue training from the previous state.

Recipes
-------

There are examples for training language models in the `recipes directory`_ for
two data sets. `penn-treebank` uses the data distributed with `RNNLM basic
examples`_. `google` uses the `WMT 2011 News Crawl data`_, processed with the
scripts provided by the `1 Billion Word Language Modeling Benchmark`_. The
examples demonstrate class-based models, hierarchical softmax, and
noise-contrastive estimation.

.. _recipes directory: https://github.com/senarvi/theanolm/tree/master/recipes
.. _RNNLM basic examples: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
.. _WMT 2011 News Crawl data: http://www.statmt.org/wmt11/translation-task.html#download
.. _1 Billion Word Language Modeling Benchmark: https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark
