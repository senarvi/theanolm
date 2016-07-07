#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from shlex import shlex
from theanolm.exceptions import InputError

class Lattice(object):
    """A word lattice that can be decoded.

    The word graph is represented as a list of nodes and links. Each node
    contains pointers to its incoming and outgoing links. Each link contains a
    pointer to the nodes in both ends.
    """

    class Link(object):
        """A link between two graph nodes.

        A link contains pointers to the start and end node. A node that has the
        link as an outgoing link can find the next node from ``end_node`` and a
        node that has the link as an incoming link can find the previous node
        from ``start_node``.
        """

        def __init__(self, start_node, end_node):
            """Constructs a link.

            :type start_node: int
            :param start_node: the node that has this link as an outgoing link

            :type end_node: int
            :param end_node: the node that has this link as an incoming link
            """

            self.start_node = start_node
            self.end_node = end_node
            self.word = None
            self.ac_score = None
            self.lm_score = None

    class Node(object):
        """A node in the graph.

        Outgoing and incoming links can be used to find the next and previous
        nodes in the topology.
        """

        def __init__(self, id):
            """Constructs a node with no links.

            :type id: int
            :param id: the ID that can be used to access the node in the node
                       list
            """

            self.id = id
            self.out_links = []
            self.in_links = []
            self.time = None

    def __init__(self):
        """Constructs an empty lattice.
        """

        self._nodes = []
        self._links = []

    def read_slf(self, lattice_file):
        """Reads SLF lattice file.

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

        self._nodes = [self.Node(id) for id in range(self._num_nodes)]
        self._links = []

        for line in lattice_file:
            fields = self._split_slf_line(line)
            if not fields:
                continue
            name, value = self._split_slf_field(fields[0])
            if name == 'I':
                self._read_slf_node(int(value), fields[1:])
            elif name == 'J':
                self._read_slf_link(int(value), fields[1:])

        if len(self._links) != self._num_links:
            raise InputError("Number of links in SLF lattice doesn't match the "
                             "LINKS field.")

        if not self._initial_node_id is None:
            self._initial_node = self._nodes[self._initial_node_id]
        else:
            # Find the node with no incoming links.
            self._initial_node = None
            for node in self._nodes:
                if len(node.in_links) == 0:
                    self._initial_node = node
                    break
            if self._initial_node is None:
                raise InputError("Could not find initial node in SLF lattice.")

        if not self._final_node_id is None:
            self._final_node = self._nodes[self._final_node_id]
        else:
            # Find the node with no outgoing links.
            self._final_node = None
            for node in self._nodes:
                if len(node.out_links) == 0:
                    self._final_node = node
                    break
            if self._final_node is None:
                raise InputError("Could not find final node in SLF lattice.")

        self._sort_nodes()

    def _read_slf_header(self, fields):
        """Reads SLF lattice header fields and saves them in member variables.

        :type fields: list of strs
        :param fields: fields, such as name="value"
        """
        
        for field in fields:
            name, value = self._split_slf_field(field)
            if (name == 'UTTERANCE') or (name == 'U'):
                self._utterance_id = value
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

        :type node_id: int
        :param node_id: ID of the node

        :type fields: list of strs
        :param fields: the rest of the node fields after ID
        """

        node = self._nodes[node_id]
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
                start_node = self._nodes[int(value)]
            elif (name == 'END') or (name == 'E'):
                end_node = self._nodes[int(value)]
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

    def _add_link(self, start_node, end_node):
        """Adds a link between two nodes.

        :type start_node: int
        :param start_node: creates a link from this node

        :type end_node: int
        :param end_node: creates a link to this node

        :rtype: Link
        :returns: the created link
        """

        link = self.Link(start_node, end_node)
        self._links.append(link)
        start_node.out_links.append(link)
        end_node.in_links.append(link)
        return link

    def _sort_nodes(self):
        """Sorts nodes topologically, then by time.

        Creates ``_sorted_nodes``, which contains the nodes in sorted order.
        Uses the Kahn's algorithm to sort the nodes topologically, but always
        picks the node from the queue that has the lowest time stamp, if the
        nodes contain time stamps.
        """

        self._sorted_nodes = []
        # A queue of nodes to be visited next:
        node_queue = [self._initial_node]
        # The number of incoming links not traversed yet:
        in_degrees = [len(node.in_links) for node in self._nodes]
        while node_queue:
            node = node_queue.pop()
            self._sorted_nodes.append(node)
            for link in node.out_links:
                next_node = link.end_node
                in_degrees[next_node.id] -= 1
                if in_degrees[next_node.id] == 0:
                    node_queue.append(next_node)
                    node_queue.sort(key=lambda x: (x.time is None, x.time),
                                    reverse=True)
                elif in_degrees[next_node.id] < 0:
                    raise InputError("Word lattice contains a cycle.")

        if len(self._sorted_nodes) < len(self._nodes):
            logging.warning("Word lattice contains unreachable nodes.")
        else:
            assert len(self._sorted_nodes) == len(self._nodes)
