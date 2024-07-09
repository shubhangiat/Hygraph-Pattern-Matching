import networkx as nx
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
class Node:
    def __init__(self, oid, label, start_time, end_time=None):
        self.oid = oid
        self.label = label
        self.membership = {}

class PGNode(Node):
    def __init__(self, oid, label, start_time, end_time=None):
        super().__init__(oid, label)
        self.start_time = start_time
        self.end_time = end_time
        self.properties = {}

class TSNode(Node):
    def __init__(self, oid, label):
        super().__init__(oid, label)
        self.series = {}
class Edge:
    def __init__(self, oid, source, target, label, start_time, end_time=None):
        self.oid = oid
        self.source = source
        self.target = target
        self.label = label
        self.start_time = start_time
        self.end_time = end_time
        self.properties = {}
        self.membership = {}

class TimeSeries:
    def __init__(self, tsid, data):
        self.tsid = tsid
        self.data = data

class HyGraph:
    def __init__(self):
        self.graph = nx.MultiGraph()
        self.time_series = {}
        self.subgraphs = {}
    def add_node(self, node):
        self.graph.add_node(node.oid, data=node)

    def add_edge(self, edge):
        if isinstance(edge.source, type(edge.target)):
            self.graph.add_edge(edge.source, edge.target, key=edge.oid, data=edge)
        else:
            raise ValueError("Source and target nodes must be of the same type.")



    def add_edge(self, oid, source, target, label, start_time, end_time=None):
        self.graph.add_edge(source, target, key=oid, label=label, start_time=start_time, end_time=end_time, properties={}, memberships={})

    def add_subgraph(self, subgraph_id, label, start_time, end_time=None):
        self.subgraphs[subgraph_id] = {'start_time': start_time, 'end_time': end_time, 'label' : label, 'properties': {}}
    def create_time_series(self, tsid, timestamps, variables, data):
        """
        Create and add a multivariate time series to the graph.
        :param tsid: Time series ID
        :param timestamps: List of timestamps
        :param variables: List of variable names
        :param data: 2D array-like structure with data
        """
        time_index = pd.to_datetime(timestamps)
        data_array = xr.DataArray(data, coords=[time_index, variables], dims=['time', 'variable'], name=f'ts_{tsid}')
        self.time_series[tsid] = TimeSeries(tsid, data_array)
    def add_property(self, element_type, oid, property_key, tsid):
            """
            Add a property to a node, edge, or subgraph.
            :param element_type: Type of the element ('node', 'edge', or 'subgraph').
            :param oid: ID of the element.
            :param property_key: Key of the property.
            :param tsid: Time series ID.
            """
            if element_type == 'node':
                if oid not in self.graph.nodes:
                    raise ValueError(f"Node with ID {oid} does not exist.")
                self.graph.nodes[oid]['properties'][property_key] = tsid

            elif element_type == 'edge':
                found = False
                for source, target, key in self.graph.edges(keys=True):
                    if key == oid:
                        self.graph.edges[source, target, key]['properties'][property_key] = tsid
                        found = True
                        break
                if not found:
                    raise ValueError(f"Edge with ID {oid} does not exist.")

            elif element_type == 'subgraph':
                if oid not in self.subgraphs:
                    raise ValueError(f"Subgraph with ID {oid} does not exist.")
                self.subgraphs[oid]['properties'][property_key] = tsid

    def add_membership(self, element_type, oid, tsid):
        """
        Add a membership time series to a node or edge.
        :param element_type: Type of the element ('node' or 'edge').
        :param oid: ID of the node or edge.
        :param tsid: Time series ID of the membership.
        """
        if element_type == 'node':
            if oid not in self.graph.nodes:
                raise ValueError(f"Node with ID {oid} does not exist.")
            self.graph.nodes[oid]['memberships'] = tsid

        elif element_type == 'edge':
            found = False
            for source, target, key in self.graph.edges(keys=True):
                if key == oid:
                    self.graph.edges[source, target, key]['memberships'] = tsid
                    found = True
                    break
            if not found:
                raise ValueError(f"Edge with ID {oid} does not exist.")


    def get_node(self, oid):
        if oid not in self.graph.nodes:
            raise ValueError(f"Node with ID {oid} does not exist.")
        return self.graph.nodes[oid]

    def get_edge(self, oid):
        if oid not in self.graph.edges:
            raise ValueError(f"Edge with ID {oid} does not exist.")
        return self.graph.edges[oid]

    def get_time_series(self, tsid):
        if tsid not in self.time_series:
            raise ValueError(f"Time series with ID {tsid} does not exist.")
        return self.time_series[tsid].data

    def get_subgraph(self, subgraph_id):
        if subgraph_id not in self.subgraphs:
            raise ValueError(f"Subgraph with ID {subgraph_id} does not exist.")
        return self.subgraphs[subgraph_id]

    def display(self):
        print("Nodes:")
        for node_id, data in self.graph.nodes(data=True):
            print(f"Node {node_id}: {data}")

        print("\nEdges:")
        for source, target, key, data in self.graph.edges(keys=True, data=True):
            print(f"Edge {key} from {source} to {target}: {data}")

        print("\nSubgraphs:")
        for subgraph_id, data in self.subgraphs.items():
            print(f"Subgraph {subgraph_id}: {data}")

        print("\nTime Series:")
        for tsid, ts in self.time_series.items():
            print(f"Time Series {tsid}:")
            variables = [str(var) for var in ts.data.coords['variable'].values]
            print(f"Variables: {', '.join(variables)}")
            ts_df = ts.data.to_dataframe('value').reset_index()
            grouped = ts_df.groupby('time')
            for time, group in grouped:
                values = [f"{row['variable']}: {row['value']}" for idx, row in group.iterrows()]
                row_str = ", ".join(values)
                print(f"{time}, {row_str}")



