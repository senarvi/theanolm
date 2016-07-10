#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from shlex import shlex
from theanolm.exceptions import InputError
from theanolm.scoring.lattice import Lattice

class SLFLattice(Lattice):
    """SLF Format Word Lattice

    A word lattice that can be read in SLF format.
    """

    def read(self, lattice_file):
        """Reads an SLF lattice file.

        :type lattice_file: file object
        :param lattice_file: a file in SLF lattice format
        """

        self._utterance_id = None
        self._log_base = None
        self._lm_scale = 1.0
        self._wi_penalty = 0.0
        self._num_nodes = None
        self._num_links = None
        self._initial_node_id = None
        self._final_node_id = None

        for line in lattice_file:
            fields = self._split_slf_line(line)
            self._read_slf_header(fields)
            if (not self._num_nodes is None) and (not self._num_links is None):
                break

        self.nodes = [self.Node(id) for id in range(self._num_nodes)]
        self.links = []

        for line in lattice_file:
            fields = self._split_slf_line(line)
            if not fields:
                continue
            name, value = self._split_slf_field(fields[0])
            if name == 'I':
                self._read_slf_node(int(value), fields[1:])
            elif name == 'J':
                self._read_slf_link(int(value), fields[1:])

        if len(self.links) != self._num_links:
            raise InputError("Number of links in SLF lattice doesn't match the "
                             "LINKS field.")

        if not self._initial_node_id is None:
            self.initial_node = self.nodes[self._initial_node_id]
        else:
            # Find the node with no incoming links.
            self.initial_node = None
            for node in self.nodes:
                if len(node.in_links) == 0:
                    self.initial_node = node
                    break
            if self.initial_node is None:
                raise InputError("Could not find initial node in SLF lattice.")

        if not self._final_node_id is None:
            self.final_node = self.nodes[self._final_node_id]
        else:
            # Find the node with no outgoing links.
            self.final_node = None
            for node in self.nodes:
                if len(node.out_links) == 0:
                    self.final_node = node
                    break
            if self.final_node is None:
                raise InputError("Could not find final node in SLF lattice.")

        # If word identity information is not present in node definitions then
        # it must appear in link definitions.
        self._move_words_to_links()
        for link in self.links:
            if link.word is None:
                raise InputError("SLF lattice does not contain word identity "
                                 "in link %d or in the following node.".format(
                                 link.id))

    def _read_slf_header(self, fields):
        """Reads SLF lattice header fields and saves them in member variables.

        :type fields: list of strs
        :param fields: fields, such as name="value"
        """
        
        for field in fields:
            name, value = self._split_slf_field(field)
            if (name == 'UTTERANCE') or (name == 'U'):
                self._utterance_id = value
            elif (name == 'SUBLAT') or (name == 'S'):
                raise InputError("Sub-lattices are not supported.")
            elif name == 'base':
                self._log_base = float(value)
            elif name == 'lmscale':
                self._lm_scale = float(value)
            elif name == 'wdpenalty':
                self._wi_penalty = float(value)
            elif name == 'start':
                self._initial_node_id = int(value)
            elif name == 'end':
                self._final_node_id = int(value)
            elif (name == 'NODES') or (name == 'N'):
                self._num_nodes = int(value)
            elif (name == 'LINKS') or (name == 'L'):
                self._num_links = int(value)

    def _read_slf_node(self, node_id, fields):
        """Reads SLF lattice node fields and saves the information in the given
        node.

        Some SLF lattices contain word identities in the nodes. They will be
        saved into the node, but later moved to the links leading to the node.

        :type node_id: int
        :param node_id: ID of the node

        :type fields: list of strs
        :param fields: the rest of the node fields after ID
        """

        node = self.nodes[node_id]
        for field in fields:
            name, value = self._split_slf_field(field)
            if (name == 'time') or (name == 't'):
                node.time = float(value)
            elif (name == 'WORD') or (name == 'W'):
                node.word = value

    def _read_slf_link(self, link_id, fields):
        """Reads SLF lattice link fields and creates such link.

        :type link_id: int
        :param link_id: ID of the link

        :type fields: list of strs
        :param fields: the rest of the link fields after ID
        """

        start_node = None
        end_node = None
        word = None
        ac_score = None
        lm_score = None

        for field in fields:
            name, value = self._split_slf_field(field)
            if (name == 'START') or (name == 'S'):
                start_node = self.nodes[int(value)]
            elif (name == 'END') or (name == 'E'):
                end_node = self.nodes[int(value)]
            elif (name == 'WORD') or (name == 'W'):
                word = value
            elif (name == 'acoustic') or (name == 'a'):
                ac_score = float(value)
            elif (name == 'language') or (name == 'l'):
                lm_score = float(value)

        if start_node is None:
            raise InputError("Start node is not specified for link %d.".format(
                             link_id))
        if end_node is None:
            raise InputError("End node is not specified for link %d.".format(
                             link_id))
        link = self._add_link(start_node, end_node)
        link.word = word
        link.ac_score = ac_score
        link.lm_score = lm_score

    def _split_slf_line(self, line):
        """Parses a list of fields from an SLF lattice line.

        Each field contains a name, followed by =, followed by a possible quoted
        value. Only double quotes can be used for quotation, and for the literal
        " the double quote must be escaped (\"). I'm not surprise if other
        implementations or the standard doesn't agree.

        :type line: str
        :param line: a line from an SLF file

        :rtype: list of strs
        :returns: list of fields found from the line, with possible quotation
                  marks removed
        """

        lex = shlex(line, posix=True)
        lex.quotes = '"'
        lex.wordchars += "'"
        lex.whitespace_split = True
        return list(lex)

    def _split_slf_field(self, field):
        """Parses the name and value from an SLF lattice field.

        :type field: str
        :param field: a field, such as 'UTTERANCE=utterance 123'

        :rtype: tuple of two strs
        :returns: the name and value of the field
        """
        
        name_value = field.split('=', 1)
        if len(name_value) != 2:
            raise InputError("Expected '=' in SLF lattice field: '%s'".format(
                             field))
        name = name_value[0]
        value = name_value[1]
        return name, value

    def _move_words_to_links(self):
        """Move word identities from nodes to the links leading to the node.

        SLF lattices may contain words either in the nodes or in the links. If
        they are in the nodes, move them to the link leading to the node. Note
        that if there's a word in the initial node, it will be discarded. This
        is what SRILM does as well, and the SLF format says that the words are
        in the end nodes in forward lattices.
        """

        visited = { self.initial_node.id }

        def visit_link(link):
            end_node = link.end_node
            if hasattr(end_node, 'word') and isinstance(end_node.word, str):
                if link.word is None:
                    link.word = end_node.word
                else:
                    raise InputError("SLF lattice contains words both in nodes "
                                     "and links.")
            if not end_node.id in visited:
                visited.add(end_node.id)
                for next_link in end_node.out_links:
                    visit_link(next_link)

        for link in self.initial_node.out_links:
            visit_link(link)

        for node in self.nodes:
            if hasattr(node, 'word'):
                del node.word
