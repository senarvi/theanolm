#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the "theanolm decode" command.
"""

import logging

from theanolm.backend import TextFileType
from theanolm.scoring.slflattice import SLFLattice
from theanolm.scoring.kaldilattice import KaldiLattice, read_kaldi_vocabulary

class LatticeBatch(object):
    def __init__(self, lattices, lattice_list_file, lattice_format,
                 kaldi_vocabulary=None, num_jobs=1, job_id=0):
        """Reads the Kaldi word ID mapping, if given, and slices the lattices
        corresponding to the given job ID.

        If there's nothing else in the lattice list, adds the standard input.

        :type lattices: list of strs
        :param lattices: a list of lattice file paths

        :type lattice_list_file: file object
        :param lattice_list_file: a file containing paths to lattice files that
                                  will be added to the lattice list

        :type lattice_format: str
        :param lattice_format: format in which the lattices are saved; either
                               ``slf`` or ``kaldi``

        :type kaldi_vocabulary: file object
        :param kaldi_vocabulary: if not ``None``, the word to ID mapping for
                                 Kaldi lattices will be read from this file

        :type num_jobs: int
        :param num_jobs: split the lattice files to this many jobs

        :type job_id: int
        :param job_id: a number between ``0`` and ``num_jobs - 1``; select the
                       lattices corresponding to this job
        """

        # Read Kaldi word ID mapping.
        if kaldi_vocabulary is not None:
            self.kaldi_word_to_id = read_kaldi_vocabulary(kaldi_vocabulary)
            self.kaldi_id_to_word = [None] * len(self.kaldi_word_to_id)
            for word, id in self.kaldi_word_to_id.items():
                self.kaldi_id_to_word[id] = word
        elif lattice_format == 'kaldi':
            raise ValueError("Kaldi lattice vocabulary is not given.")

        if (lattice_format != 'slf') and (lattice_format != 'kaldi'):
            raise ValueError("Invalid lattice format specified ({})."
                             .format(lattice_format))
        self._lattice_format = lattice_format

        # Combine paths from command line and lattice list.
        if lattice_list_file is not None:
            lattices.extend(lattice_list_file.readlines())
        lattices = [path.strip() for path in lattices]
        # Ignore empty lines in the lattice list.
        lattices = [x for x in lattices if x]
        if not lattices:
            lattices = ["-"]

        # Pick every ith lattice, if --num-jobs is specified and > 1.
        if num_jobs < 1:
            raise ValueError("Invalid number of jobs specified ({})."
                             .format(num_jobs))
        if (job_id < 0) or (job_id > num_jobs - 1):
            raise ValueError("Invalid job selected ({})."
                             .format(job_id))
        self._lattices = lattices[job_id::num_jobs]

    def __iter__(self):
        """A generator for iterating through the lattices of this job.
        """

        file_type = TextFileType('r')

        for path in self._lattices:
            logging.info("Reading lattice file `%s´.", path)
            lattice_file = file_type(path)
            if self._lattice_format == 'slf':
                yield SLFLattice(lattice_file)
            else:
                assert self._lattice_format == 'kaldi'
                lattice_lines = []
                id_to_word = self.kaldi_id_to_word
                while True:
                    line = lattice_file.readline()
                    if not line:
                        # end of file
                        if lattice_lines:
                            yield KaldiLattice(lattice_lines, id_to_word)
                        break
                    line = line.strip()
                    if not line:
                        # empty line
                        if lattice_lines:
                            yield KaldiLattice(lattice_lines, id_to_word)
                        lattice_lines = []
                        continue
                    lattice_lines.append(line)
