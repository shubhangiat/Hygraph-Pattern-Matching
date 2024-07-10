import networkx as nx
import pandas as pd
import xarray as xr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
from igraph import Graph as IGraph
from IDGenerator import IDGenerator
from Oberserver import Subject


class Node(Subject):
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
    def __init__(self, oid, label,time_series):
        super().__init__(oid, label)
        self.series = time_series
class Edge(Subject):
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
    """
     Create and add a multivariate time series to the graph.
     :param tsid: Time series ID
     :param timestamps: List of timestamps
     :param variables: List of variable names
     :param data: 2D array-like structure with data
     """
    def __init__(self, tsid, timestamps, variables, data):
        self.tsid = tsid
        time_index = pd.to_datetime(timestamps)
        self.data = xr.DataArray(data, coords=[time_index, variables], dims=['time', 'variable'], name=f'ts_{tsid}')

class Subgraph(Subject):
    def __init__(self, subgraph_id, label, start_time, end_time=None, filter_func=None):
        self.subgraph_id = subgraph_id
        self.label = label
        self.start_time = start_time
        self.end_time = end_time
        self.properties = {}
        self.filter_func = filter_func


class HyGraph:
    def __init__(self):
        self.graph = nx.MultiGraph()
        self.time_series = {}
        self.subgraphs = {}
        self.id_generator = IDGenerator()
    def add_node(self, node):
        self.graph.add_node(node.oid, data=node)

    def add_edge(self, edge):
        self.graph.add_edge(edge.source, edge.target, key=edge.oid, data=edge)


    def add_subgraph(self, subgraph):
        subgraph_view = nx.subgraph_view(self.graph, filter_node=subgraph.filter_func, filter_edge=subgraph.filter_func)
        self.subgraphs[subgraph.subgraph_id] = {'view': subgraph_view, 'data': subgraph}


    def get_element(self, element_type, oid):
        if element_type == 'node':
            if oid not in self.graph.nodes:
                raise ValueError(f"Node with ID {oid} does not exist.")
            return self.graph.nodes[oid]['data']
        elif element_type == 'edge':
            for source, target, key in self.graph.edges(keys=True):
                if key == oid:
                    return self.graph.edges[source, target, key]['data']
            raise ValueError(f"Edge with ID {oid} does not exist.")
        elif element_type == 'subgraph':
            if oid not in self.subgraphs:
                raise ValueError(f"Subgraph with ID {oid} does not exist.")
            return self.subgraphs[oid]['data']


    def add_property(self, element_type, oid, property_key, value):
        element = self.get_element(element_type, oid)
        element.properties[property_key] = value
        element.notify()
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

    def create_similarity_edges(self, similarity_threshold):
        ts_nodes = [node for node in self.graph.nodes(data=True) if isinstance(node[1]['data'], TSNode)]
        edge_id = len(self.graph.edges)

        for i in range(len(ts_nodes)):
            for j in range(i + 1, len(ts_nodes)):
                ts1 = ts_nodes[i][1]['data'].series.data.values
                ts2 = ts_nodes[j][1]['data'].series.data.values
                distance, _ = fastdtw(ts1, ts2, dist=euclidean)

                if distance <= similarity_threshold:
                    start_time = datetime.now()
                    edge = Edge(oid=edge_id, source=ts_nodes[i][1]['data'], target=ts_nodes[j][1]['data'], label='similarTo', start_time=start_time)
                    self.add_edge(edge)

                    # Calculate and add the degree of similarity over time as a property
                    similarity_over_time = [distance] * len(ts1)
                    timestamps = pd.date_range(start=start_time, periods=len(similarity_over_time), freq='D')
                    tsid = f"similarity_{edge_id}"
                    time_series = TimeSeries(tsid, timestamps, ['similarity'], [similarity_over_time])
                    self.time_series[tsid] = time_series
                    self.add_property('edge', edge_id, 'degree_similarity_over_time', tsid)

                    edge_id += 1
    def graph_metrics_evolution(self):
        igraph_g = IGraph.TupleList(self.graph.edges(), directed=False)
        igraph_g.vs["name"] = list(self.graph.nodes())

        communities = igraph_g.community_infomap()

        timestamp = datetime.now()
        for idx, community in enumerate(communities):
            for node in community:
                node_id = igraph_g.vs[node]["name"]
                self.graph.nodes[node_id]['data'].memberships[timestamp] = idx

        for edge in self.graph.edges(data=True):
            source, target, key = edge
            source_community = self.graph.nodes[source]['data'].memberships[timestamp]
            target_community = self.graph.nodes[target]['data'].memberships[timestamp]
            if source_community == target_community:
                self.graph.edges[source, target, key]['data'].memberships[timestamp] = source_community
            else:
                self.graph.edges[source, target, key]['data'].memberships[timestamp] = f"{source_community},{target_community}"


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



