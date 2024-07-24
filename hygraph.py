import networkx as nx
import numpy as np
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
    def __init__(self, oid, label):
        super().__init__()  # Initialize the Subject
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
        super().__init__()
        self.oid = oid
        self.source = source
        self.target = target
        self.label = label
        self.start_time = start_time
        self.end_time = end_time
        self.properties = {}
        self.membership = {}

class TimeSeriesMetadata:
    def __init__(self, owner_id, edge_label='', element_type='', attribute=''):
        self.owner_id = owner_id
        self.edge_label = edge_label
        self.element_type = element_type
        self.attribute = attribute

class TimeSeries:
    """
     Create and add a multivariate time series to the graph.
     :param tsid: Time series ID
     :param timestamps: List of timestamps
     :param variables: List of variable names
     :param data: 2D array-like structure with data
     """
    def __init__(self, tsid, timestamps, variables, data, metadata=None):
        self.tsid = tsid
        time_index = pd.to_datetime(timestamps)
        self.data = xr.DataArray(data, coords=[time_index, variables], dims=['time', 'variable'], name=f'ts_{tsid}')
        self.metadata = metadata if metadata is not None else {}

    def append_data(self, date, value):
        date = pd.to_datetime(date)
        new_data = xr.DataArray([[value]], coords=[[date], self.data.coords['variable']], dims=['time', 'variable'])
        self.data = xr.concat([self.data, new_data], dim='time')

class Subgraph(Subject):
    def __init__(self, subgraph_id, label, start_time, end_time=None, filter_func=None):
        super().__init__()
        self.subgraph_id = subgraph_id
        self.label = label
        self.start_time = start_time
        self.end_time = end_time
        self.properties = {}
        self.filter_func = filter_func


class GraphObserver:
    def __init__(self, hygraph):
        self.hygraph = hygraph
        self.currently_updating = set()

    def update(self, subject):
        print(f"GraphObserver: Update called for {type(subject).__name__} with ID {subject.oid}")
        if subject not in self.currently_updating:
            self.currently_updating.add(subject)
            element_type = 'node' if isinstance(subject, Node) else 'edge'
            for metric in self.hygraph.custom_metrics:
                if metric['element_type'] != element_type:
                    continue  # Skip if the element type does not match
                if metric.get('label') and subject.label != metric['label']:
                    continue  # Skip if the label does not match
                oid = subject.oid
                attribute = metric['attribute']
                current_value = metric['aggregate_function'](
                    self.hygraph,
                    element_type,
                    oid,
                    attribute,
                    datetime.now()
                )
                if attribute in subject.properties:
                    print(f"Updating time series for {oid}, attribute {attribute}")
                    tsid = subject.properties[attribute]
                    self.hygraph.append_time_series(tsid, datetime.now(), current_value)
                else:
                    print(f"Creating time series for {oid}, attribute {attribute}")
                    self.hygraph.create_time_series_from_graph(
                        element_type=element_type,
                        oid=oid,
                        attribute=attribute,
                        start_date=subject.start_time,
                        end_date=None,
                        aggregate_function=metric['aggregate_function'],
                        edge_label=subject.label if isinstance(subject, Edge) else None
                    )
            self.currently_updating.remove(subject)

class HyGraph:
    def __init__(self):
        self.graph = nx.MultiGraph()
        self.time_series = {}
        self.subgraphs = {}
        self.id_generator = IDGenerator()
        self.graph_observer = GraphObserver(self)
        self.custom_metrics = []

    def add_node(self, node):
        self.graph.add_node(node.oid, data=node)
        node.attach(self.graph_observer)
        node.notify()

    def add_edge(self, edge):
        self.graph.add_edge(edge.source, edge.target, key=edge.oid, data=edge)
        edge.attach(self.graph_observer)
        edge.notify()

    def add_subgraph(self, subgraph):
        subgraph_view = nx.subgraph_view(self.graph, filter_node=subgraph.filter_func, filter_edge=subgraph.filter_func)
        self.subgraphs[subgraph.subgraph_id] = {'view': subgraph_view, 'data': subgraph}
        subgraph.attach(self.graph_observer)


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
        print(f"addProperty {element.properties}")


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
        edge_id = self.id_generator.generate_edge_id()

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
                    tsid = self.id_generator.generate_timeseries_id()
                    time_series = TimeSeries(tsid, timestamps, ['similarity'], [similarity_over_time])
                    self.time_series[tsid] = time_series
                    self.add_property('edge', edge.oid, 'degree_similarity_over_time', tsid)

    def create_time_series_from_graph(self, element_type, oid, attribute, start_date, aggregate_function, end_date=None, freq='D', edge_label=None):
        """
        Generic function to create a time series from the graph.
        :param edge_label:
        :param element_type: Type of the element ('node', 'edge', or 'subgraph').
        :param oid: ID of the element.
        :param attribute: Attribute to create the time series for.
        :param start_date: Start date for the time series.
        :param end_date: End date for the time series.
        :param freq: Frequency of the time series (default is daily).
        :param aggregate_function: User-defined function to aggregate data.
        """
        if aggregate_function is None:
            raise ValueError("An aggregate_function must be provided.")
        if end_date is None:
            end_date = datetime.now()
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        values = []
        last_value = None
        for date in date_range:
            current_value = aggregate_function(self, element_type, oid, attribute, date)
            if current_value != last_value:
                values.append((date, current_value))
                last_value = current_value

        if values:
            timestamps, data_values = zip(*values)
            reshaped_data_values = np.array(data_values)[:, np.newaxis]

            tsid = self.id_generator.generate_timeseries_id()
            metadata = TimeSeriesMetadata(oid, edge_label if edge_label else '', element_type, attribute)
            time_series = TimeSeries(tsid, timestamps, [attribute], reshaped_data_values, metadata)
            self.time_series[tsid] = time_series
            self.add_property(element_type, oid, attribute, tsid)

    def graph_metrics_evolution(self):
        igraph_g = IGraph.TupleList(self.graph.edges(), directed=False)
        igraph_g.vs["name"] = list(self.graph.nodes())

        communities = igraph_g.community_infomap()

        timestamp = datetime.now()
        for idx, community in enumerate(communities):
            for node in community:
                node_id = igraph_g.vs[node]["name"]
                self.graph.nodes[node_id]['data'].membership[timestamp] = idx

        for source, target, key in self.graph.edges(keys=True):
            source_community = self.graph.nodes[source]['data'].membership.get(timestamp)
            target_community = self.graph.nodes[target]['data'].membership.get(timestamp)
            if source_community == target_community:
                self.graph.edges[source, target, key]['data'].membership[timestamp] = source_community
            else:
                self.graph.edges[source, target, key]['data'].membership[timestamp] = f"{source_community},{target_community}"

    def register_custom_metric(self, element_type, attribute, aggregate_function, label=None):
        self.custom_metrics.append({
            'element_type': element_type,
            'attribute': attribute,
            'aggregate_function': aggregate_function,
            'label': label
        })

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
            print(f"Time Series {tsid}: {ts.metadata.owner_id}")
            variables = [str(var) for var in ts.data.coords['variable'].values]
            print(f"Variables: {', '.join(variables)}")
            ts_df = ts.data.to_dataframe('value').reset_index()
            grouped = ts_df.groupby('time')
            for time, group in grouped:
                values = [f"{row['variable']}: {row['value']}" for idx, row in group.iterrows()]
                row_str = ", ".join(values)
                print(f"{time}, {row_str}")