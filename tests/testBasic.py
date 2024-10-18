import unittest
import numpy as np
from datetime import datetime
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../hygraph_core')))
from hygraph_core.hygraph import HyGraph


class TestHyGraph(unittest.TestCase):

    def setUp(self):
        """
        Set up the HyGraph instance and some basic nodes and edges for each test case.
        """
        self.graph = HyGraph()
        self.oid_pg_node = 1
        self.oid_ts_node = 2
        self.oid_pg_edge = 3
        self.oid_ts_edge = 4

    def test_add_pg_node(self):
        """ Test adding a PGNode to the graph with properties """
        properties = {"type": "person", "age": 30}
        pg_node = self.graph.add_pg_node(self.oid_pg_node, "PG Node", datetime.now(), properties=properties)
        self.assertEqual(pg_node.oid, self.oid_pg_node)
        self.assertEqual(pg_node.label, "PG Node")
        self.assertEqual(pg_node.properties["type"], "person")
        self.assertEqual(pg_node.properties["age"], 30)
        self.assertIn(self.oid_pg_node, self.graph.graph.nodes)

    def test_add_ts_node(self):
        """ Test adding a TSNode to the graph """
        ts_data = np.array([1.0, 2.0, 3.0])
        ts_node = self.graph.add_ts_node(self.oid_ts_node, "TS Node", ts_data)
        self.assertEqual(ts_node.oid, self.oid_ts_node)
        self.assertEqual(ts_node.label, "TS Node")
        self.assertTrue(np.array_equal(ts_node.series, ts_data))
        self.assertIn(self.oid_ts_node, self.graph.graph.nodes)

    def test_remove_node(self):
        """ Test removing a node from the graph """
        self.graph.add_pg_node(self.oid_pg_node, "PG Node", datetime.now())
        self.graph.remove_node(self.oid_pg_node)
        self.assertNotIn(self.oid_pg_node, self.graph.graph.nodes)

    def test_update_node_properties(self):
        """ Test updating node properties """
        self.graph.add_pg_node(self.oid_pg_node, "PG Node", datetime.now())
        new_properties = {"status": "active"}
        self.graph.update_node_properties(self.oid_pg_node, new_properties)
        node = self.graph.graph.nodes[self.oid_pg_node]['data']
        self.assertEqual(node.properties["status"], "active")

    def test_get_node_property(self):
        """ Test retrieving a property from a node """
        self.graph.add_pg_node(self.oid_pg_node, "PG Node", datetime.now(), properties={"status": "inactive"})
        status = self.graph.get_node_property(self.oid_pg_node, "status")
        self.assertEqual(status, "inactive")

    def test_set_node_property(self):
        """ Test setting or updating a property for a node """
        self.graph.add_pg_node(self.oid_pg_node, "PG Node", datetime.now())
        self.graph.set_node_property(self.oid_pg_node, "role", "admin")
        node = self.graph.graph.nodes[self.oid_pg_node]['data']
        self.assertEqual(node.properties["role"], "admin")

    def test_add_pg_edge(self):
        """ Test adding a PGEdge to the graph with properties """
        # Add nodes first before adding the edge
        self.graph.add_pg_node(self.oid_pg_node, "PG Node", datetime.now())
        self.graph.add_pg_node(self.oid_ts_node, "Another Node", datetime.now())

        properties = {"weight": 1.5, "relationship": "friend"}

        # add the edge
        pg_edge = self.graph.add_pg_edge(self.oid_pg_edge, self.oid_pg_node, self.oid_ts_node, "connects",
                                         datetime.now(), properties=properties)

        # Check for properties and the presence of the full edge tuple in the graph
        self.assertEqual(pg_edge.properties["weight"], 1.5)
        self.assertEqual(pg_edge.properties["relationship"], "friend")
        self.assertIn((self.oid_pg_node, self.oid_ts_node, self.oid_pg_edge),
                      self.graph.graph.edges(keys=True))

    def test_add_ts_edge(self):
        """ Test adding a TSEdge to the graph """
        # Add nodes first before adding the edge
        self.graph.add_pg_node(self.oid_pg_node, "PG Node", datetime.now())
        self.graph.add_pg_node(self.oid_ts_node, "Another Node", datetime.now())

        ts_data = np.array([1.0, 2.0, 3.0])

        # add the edge
        ts_edge = self.graph.add_ts_edge(self.oid_ts_edge, self.oid_pg_node, self.oid_ts_node, "measures", ts_data,
                                         datetime.now())

        # Check for the presence of the full edge tuple in the graph
        self.assertIn((self.oid_pg_node, self.oid_ts_node, self.oid_ts_edge),
                      self.graph.graph.edges(keys=True))

    def test_remove_edge(self):
        """ Test removing an edge from the graph """
        # Add nodes and an edge first
        self.graph.add_pg_node(self.oid_pg_node, "PG Node", datetime.now())
        self.graph.add_pg_node(self.oid_ts_node, "Another Node", datetime.now())
        self.graph.add_pg_edge(self.oid_pg_edge, self.oid_pg_node, self.oid_ts_node, "connects", datetime.now())

        # remove the edge
        self.graph.remove_edge(self.oid_pg_edge)
        edge_data = list(self.graph.graph.edges(self.oid_pg_node, self.oid_ts_node, keys=True))
        self.assertNotIn(self.oid_pg_edge, edge_data)

    def test_update_edge_properties(self):
        """ Test updating edge properties """
        # Add nodes and an edge first
        self.graph.add_pg_node(self.oid_pg_node, "PG Node", datetime.now())
        self.graph.add_pg_node(self.oid_ts_node, "Another Node", datetime.now())
        self.graph.add_pg_edge(self.oid_pg_edge, self.oid_pg_node, self.oid_ts_node, "connects", datetime.now())

        # Update properties of the edge
        new_properties = {"weight": 0.75}
        self.graph.update_edge_properties(self.oid_pg_edge, new_properties)
        edge = self.graph.graph[self.oid_pg_node][self.oid_ts_node][self.oid_pg_edge]['data']
        self.assertEqual(edge.properties["weight"], 0.75)

    def test_get_edge_property(self):
        """ Test retrieving a property from an edge """
        # Add nodes and an edge with properties
        self.graph.add_pg_node(self.oid_pg_node, "PG Node", datetime.now())
        self.graph.add_pg_node(self.oid_ts_node, "Another Node", datetime.now())
        self.graph.add_pg_edge(self.oid_pg_edge, self.oid_pg_node, self.oid_ts_node, "connects", datetime.now(),
                               properties={"distance": 100})

        # Retrieve and assert the property value
        distance = self.graph.get_edge_property(self.oid_pg_edge, "distance")
        self.assertEqual(distance, 100)

    def test_set_edge_property(self):
        """ Test setting or updating a property for an edge """
        # Add nodes and an edge first
        self.graph.add_pg_node(self.oid_pg_node, "PG Node", datetime.now())
        self.graph.add_pg_node(self.oid_ts_node, "Another Node", datetime.now())
        self.graph.add_pg_edge(self.oid_pg_edge, self.oid_pg_node, self.oid_ts_node, "connects", datetime.now())

        # Set a property on the edge
        self.graph.set_edge_property(self.oid_pg_edge, "length", 50)
        edge = self.graph.graph[self.oid_pg_node][self.oid_ts_node][self.oid_pg_edge]['data']
        self.assertEqual(edge.properties["length"], 50)

    def test_get_node_type(self):
        """ Test retrieving the type of nodes """
        pg_node = self.graph.add_pg_node(self.oid_pg_node, "PG Node", datetime.now())
        ts_node = self.graph.add_ts_node(self.oid_ts_node, "TS Node", np.array([1.0, 2.0]))

        # Check node types
        node_type_pg = pg_node.get_type()
        node_type_ts = ts_node.get_type()

        self.assertEqual(node_type_pg, "PGNode")
        self.assertEqual(node_type_ts, "TSNode")


if __name__ == '__main__':
    unittest.main()
