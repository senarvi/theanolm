#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import logging
from theanolm.exceptions import InputError

class Lattice(object):
    """Word Lattice

    Word lattice describes a search space for decoding. The graph is represented
    as a list of nodes and links. Each node contains pointers to its incoming
    and outgoing links. Each link contains a pointer to the nodes in both ends.
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

    __metaclass__ = ABCMeta

    def __init__(self):
        """Constructs an empty lattice.
        """

        self.nodes = []
        self.links = []
        self.initial_node = None
        self.final_node = None

    @abstractmethod
    def read(self, lattice_file):
        """Reads a lattice file.

        Has to be implemented by the subclass.

        :type lattice_file: file object
        :param lattice_file: a lattice file in the correct format
        """

        return

    def sorted_nodes(self):
        """Sorts nodes topologically, then by time.

        Returns a list which contains the nodes in sorted order. Uses the Kahn's
        algorithm to sort the nodes topologically, but always picks the node
        from the queue that has the lowest time stamp, if the nodes contain time
        stamps.
        """

        result = []
        # A queue of nodes to be visited next:
        node_queue = [self.initial_node]
        # The number of incoming links not traversed yet:
        in_degrees = [len(node.in_links) for node in self.nodes]
        while node_queue:
            node = node_queue.pop()
            result.append(node)
            for link in node.out_links:
                next_node = link.end_node
                in_degrees[next_node.id] -= 1
                if in_degrees[next_node.id] == 0:
                    node_queue.append(next_node)
                    node_queue.sort(key=lambda x: (x.time is None, x.time),
                                    reverse=True)
                elif in_degrees[next_node.id] < 0:
                    raise InputError("Word lattice contains a cycle.")

        if len(result) < len(self.nodes):
            logging.warning("Word lattice contains unreachable nodes.")
        else:
            assert len(result) == len(self.nodes)

        return result

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
        self.links.append(link)
        start_node.out_links.append(link)
        end_node.in_links.append(link)
        return link
