Installation
============

pip
---

TheanoLM is available from the Python Package Index. The easiest way to install
it is using pip. It requires NumPy, SciPy, Theano, and H5py packages. Theano
requires also Six and Nose. pip tries to install all the dependencies
automatically. Notice that TheanoLM supports only Python 3. In some systems a
different version of pip is used to install Python 3 packages. In Ubuntu the
command is ``pip3``. To install system-wide, use::

    sudo pip3 install TheanoLM

Linux
-----

Linux distributions commonly provide most of the dependencies through their
package repositories. You might want to keep the dependencies up to date using
the system package manager. For example, you can install the dependencies
(except Theano) in Ubuntu by issuing the following commands, before installing
TheanoLM::

    sudo apt-get install python3-numpy python3-scipy python3-h5py
    sudo apt-get install python3-six python3-nose python3-nose-parameterized
    sudo apt-get install python3-pip

Anaconda
--------

Another easy way to install all the dependencies (except Theano) is to install
`Anaconda3 <https://www.continuum.io/downloads>`_.

Source Code
-----------

The source code is distributed through `GitHub
<https://github.com/senarvi/theanolm/>`_. If you want to develop TheanoLM, it is
convenient to work on a Git repository tree. I would recommend forking the
repository first on GitHub, so that you can commit changes to your personal copy
of the repository. Then install Theano and H5py for example using pip. Clone the
forked repository to, say, ``$HOME/git/theanolm``, and add that directory to
``$PYTHONPATH`` and the ``bin`` subdirectory to ``$PATH``::

    mkdir -p "$HOME/git"
    cd "$HOME/git"
    git clone https://github.com/my-username/theanolm.git
    export PYTHONPATH="$PYTHONPATH:$HOME/git/theanolm"
    export PATH="$PATH:$HOME/git/theanolm/bin"
