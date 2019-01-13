Installation
============

pip
---

TheanoLM is available from the Python Package Index. The easiest way to install
it is using pip. It requires NumPy, Theano, and H5py packages. Theano requires
also Six and Nose. pip tries to install all the dependencies automatically.
Notice that TheanoLM supports only Python 3. In some systems a different version
of pip is used to install Python 3 packages. In Ubuntu the command is ``pip3``.
To install system-wide, use::

    sudo pip3 install TheanoLM

This requires that you have superuser privileges. You can also install TheanoLM
and the dependencies under your home directory by passing the ``--user``
argument to pip::

    pip3 install --user TheanoLM

Installing all Python modules into a single environment can make managing the
packages and dependencies somewhat difficult. A convenient alternative is to
install TheanoLM in an isolated Python environment created using `virtualenv`_
or the standard library module venv that is shipped with Python 3.3 and later.
You have to install wheel before installing other packages. For example, to
create a virtual environment for TheanoLM in ``~/theanolm``, use::

    python3 -m venv ~/theanolm
    source ~/theanolm/bin/activate
    pip3 install wheel
    pip3 install TheanoLM

You'll continue working in the virtual environment until you deactivate it with
the ``deactivate`` command. It can be activated again using ``source
~/theanolm/bin/activate``.

Anaconda
--------

Perhaps a more convenient tool for managing a large collection of Python
packages is the conda package manager. `Anaconda`_ distribution includes the
package manager and a number of packages for scientific computing. Make sure to
select the Python 3 version.

Most of the dependencies are included in the distribution. Currently the
bleeding-edge version of Theano is required. It can be installed through the
mila-udem channel. It is also very easy to install the libgpuarray and pygpu
dependencies, required for GPU computation, in the same way::

    conda install -c mila-udem/label/pre theano pygpu libgpuarray

TheanoLM can be installed through the conda-forge channel::

    conda install -c conda-forge TheanoLM 

Linux
-----

Linux distributions commonly provide most of the dependencies through their
package repositories. You might want to keep the dependencies up to date using
the system package manager. For example, you can install the dependencies
(except Theano) in Ubuntu by issuing the following commands, before installing
TheanoLM::

    sudo apt-get install python3-numpy python3-h5py
    sudo apt-get install python3-six python3-nose python3-nose-parameterized
    sudo apt-get install python3-pip

Source Code
-----------

The source code is distributed through `GitHub
<https://github.com/senarvi/theanolm/>`_. For developing TheanoLM you need to
work on the Git repository tree. The package can be installed using pip from the
repository root. There is a convenient option ``--editable`` that causes pip to
install stub scripts that call the program binaries from the repository. This
way you avoid having to reinstall the project after every change.

I recommend forking the repository first on GitHub, so that you can commit
changes to your personal copy of the repository. Then install Theano and H5py
using pip or Anaconda. Clone the forked repository and run pip from the
repository root::

    git clone https://github.com/my-username/theanolm.git
    cd theanolm
    pip3 install --editable .

.. _virtualenv: https://virtualenv.pypa.io/en/stable/
.. _Anaconda: https://www.continuum.io/downloads
