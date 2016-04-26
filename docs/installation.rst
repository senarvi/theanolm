Installation
============

TheanoLM is available from the Python Package Index. The easiest way to install
it is using pip. It requires NumPy, SciPy, Theano, and H5py packages, and Theano
requires also Six and Nose. Although pip tries to install the dependencies as
well, it's probably better to use the system package manager to install the
packages that are provided in the system package repository. Notice that
TheanoLM supports only Python 3. In some systems a different version of pip is
used to install Python 3 packages. For example, the following commands would
install the dependencies, the correct version of pip, and TheanoLM, in Ubuntu::

    sudo apt-get install python3-numpy python3-scipy python3-h5py
    sudo apt-get install python3-six python3-nose python3-nose-parameterized
    sudo apt-get install python3-pip
    sudo pip3 install TheanoLM

If you want to develop TheanoLM, it is convenient to use it from a Git
repository tree. First make sure that you have Theano and h5py (python-h5py
Ubuntu package) installed. Clone TheanoLM Git repository to, say,
``$HOME/git/theanolm``, and add that directory to ``$PYTHONPATH`` and the
``bin`` subdirectory to ``$PATH``::

    mkdir -p "$HOME/git"
    cd "$HOME/git"
    git clone https://github.com/senarvi/theanolm.git
    export PYTHONPATH="$PYTHONPATH:$HOME/git/theanolm"
    export PATH="$PATH:$HOME/git/theanolm/bin"
