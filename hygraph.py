import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from datetime import datetime, timedelta
from igraph import Graph as IGraph
from idGenerator import IDGenerator
from oberserver import Subject


class Node(Subject):
    def __init__(self, oid, label, node_id=None):
        super().__init__()  # Initialize the Subject
        self.oid = oid
        self.node_id = node_id  # External ID from CSV file
        self.label = label
        self.membership = {}
    def __repr__(self):
        membership_str = ', '.join(f"{k}: {v}" for k, v in self.membership.items())
        return f"Node(oid={self.oid}, label={self.label}, membership={{ {membership_str} }})"
class PGNode(Node):
    def __init__(self, oid, label, start_time, end_time=None, node_id=None):
        super().__init__(oid, label,node_id)
        self.start_time = start_time
        self.end_time = end_time
        self.properties = {}
    def __repr__(self):
        properties_str = ', '.join(f"{k}: {v}" for k, v in self.properties.items())
        base_str = super().__repr__()
        return f"{base_str}, start_time={self.start_time}, end_time={self.end_time}, properties={{ {properties_str} }}"

class TSNode(Node):
    def __init__(self, oid, label,time_series):
        super().__init__(oid, label)
        self.series = time_series
    def __repr__(self):
        base_str = super().__repr__()
        return f"{base_str}, series={self.series}"
class Edge(Subject):
    def __init__(self, oid, source, target, label, start_time, end_time=None, edge_id=None):
        super().__init__()
        self.oid = oid
        self.source = source
        self.target = target
        self.label = label
        self.edge_id = edge_id  # External ID from CSV file
        self.start_time = start_time
        self.end_time = end_time
        self.properties = {}
        self.membership = {}
    def __repr__(self):
        properties_str = ', '.join(f"{k}: {v}" for k, v in self.properties.items())
        return f"Edge(oid={self.oid}, source={self.source}, target={self.target}, label={self.label}, start_time={self.start_time}, end_time={self.end_time}, properties={{ {properties_str} }})"

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

    def update_queries(self, queries):
        self.hygraph.queries = queries
        print(f"Queries updated: {queries}")

    def update_node_time_series(self, node, query, edge=None):
        attribute = query['time_series_config']['attribute']
        aggregate_function = query['time_series_config']['aggregate_function']

        current_value = aggregate_function(
            self.hygraph,
            'node',
            node.oid,
            attribute,
            datetime.now()
        )

        tsid = node.properties.get(attribute)
        if tsid and tsid in self.hygraph.time_series:
            print(f"Updating time series for {node.oid}, attribute {attribute}")
            self.hygraph.append_time_series(tsid, datetime.now(), current_value)
        else:
            print(f"Creating time series for {node.oid}, attribute {attribute}")
            self.hygraph.create_time_series_from_graph(query)

    def update(self, subject):
        print(f"GraphObserver: Update called for {type(subject).__name__} with ID {subject.oid}")

        if subject not in self.currently_updating:
            self.currently_updating.add(subject)
            element_type = "Edge"
            print('ddfdcdcd',element_type,self.hygraph.queries)
            for query in self.hygraph.queries:
                print(f"GraphObserver: in for loop {self.hygraph.queries} {query}")
                if type(subject).__name__ != element_type:
                    print("Skipping metric because element type is not 'edge'")
                    continue  # Skip if the query's element type does not match

                if query.get('label') and subject.label != query['label']:
                    print(f"Skipping metric because label does not match: {query['label']} vs {subject.label}")
                    continue  # Skip if the edge label does not match

                direction = query['time_series_config'].get('direction', 'both')
                source_node = self.hygraph.get_element('node', subject.source)
                target_node = self.hygraph.get_element('node', subject.target)

                if direction in ['in', 'both']:
                    self.update_node_time_series(target_node, query, subject)
                if direction in ['out', 'both']:
                    self.update_node_time_series(source_node, query, subject)
            self.currently_updating.remove(subject)


class HyGraph:
    def __init__(self):
        self.graph = nx.MultiGraph()
        self.time_series = {}
        self.subgraphs = {}
        self.id_generator = IDGenerator()
        self.graph_observer = GraphObserver(self)
        self.updated = False  # Flag to track updates
        self.queries = []

    def set_updated(self, value=True):
        self.updated = value
        print(f"Set updated called: {value}")

    def add_node(self, node):
        self.graph.add_node(node.oid, data=node)
        node.attach(self.graph_observer)
        self.set_updated()
        node.notify()

    def add_edge(self, edge):
        self.graph.add_edge(edge.source, edge.target, key=edge.oid, data=edge)
        print("Adding edge with:", edge.label, edge.start_time)
        edge.attach(self.graph_observer)
        self.set_updated()
        edge.notify()

    def add_subgraph(self, subgraph):
        subgraph_view = nx.subgraph_view(self.graph, filter_node=subgraph.filter_func, filter_edge=subgraph.filter_func)
        self.subgraphs[subgraph.subgraph_id] = {'view': subgraph_view, 'data': subgraph}
        subgraph.attach(self.graph_observer)

    def delete_node(self, oid, end_time=None):
        if oid not in self.graph.nodes:
            raise ValueError(f"Node with ID {oid} does not exist.")

        node = self.get_element('node', oid)
        end_time = end_time or datetime.now()
        node.end_time = end_time

        # Notify observers
        self.set_updated()
        node.notify()

    def delete_edge(self, oid, end_time=None):
        edge = None
        for u, v, key in self.graph.edges(keys=True):
            if key == oid:
                edge = self.graph.edges[u, v, key]['data']
                break

        if edge is None:
            raise ValueError(f"Edge with ID {oid} does not exist.")

        end_time = end_time or datetime.now()
        edge.end_time = end_time

        # Notify observers
        self.set_updated()
        edge.notify()

    def add_query(self, query):
        """
        Add a new query to the HyGraph instance.

        Parameters:
        - query (dict): A dictionary containing filters and configuration for time series creation.
        """
        self.queries.append(query)
        print(f"Query added: {query}")
        # If needed, you can reinitialize the observer or notify it about the new query
        self.graph_observer.update_queries(self.queries)

    def set_query(self, query):
        """
        Sets or updates the query for the HyGraph instance.
        This query will be used by the observer to generate time series.
        """
        self.query = query
        print("Query has been set/updated.")

    def get_query(self, index=None):
        """
        Returns the current query or queries being used by the HyGraph instance.
        If an index is provided, return the specific query at that index.
        """
        if index is not None:
            return self.queries[index] if 0 <= index < len(self.queries) else None
        return self.queries[len(self.queries)]

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
        if property_key not in element.properties:
            element.properties[property_key] = value
            print(f"addProperty {element.properties}")
        else:
            print(f"Property {property_key} already exists for element with ID {oid}.")

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

    def append_time_series(self, tsid, date, value):
        if tsid in self.time_series:
            self.time_series[tsid].append_data(date, value)
        else:
            raise ValueError(f"Time series with ID {tsid} does not exist.")
    def create_time_series_from_graph(self, query):
        """
    Create time series data for nodes, edges, or subgraphs based on a provided query.

    Parameters:
    - query (dict): A dictionary containing filters and configuration for time series creation.
        - element_type (str): 'node', 'edge', or 'subgraph'.
        - subgraph_id (str, optional): The specific subgraph ID to process.
        - subgraph_label (str, optional): The label to match subgraphs.
        - node_filter (function, optional): A function to filter nodes.
        - edge_filter (function, optional): A function to filter edges.
        - time_series_config (dict): Configuration for time series creation, including:
            - start_date (datetime): The start date for the time series.
            - end_date (datetime, optional): The end date for the time series.
            - attribute (str): The attribute to track in the time series.
            - aggregate_function (function): The function to aggregate values.
            - direction (str, optional): Direction for edge aggregation ('in', 'out', 'both').
            - freq (str): The frequency for the time series (e.g., 'D', 'M').
    """
        element_type = query.get('element_type', 'node')
        ts_config = query.get('time_series_config')

        if element_type == 'node':
            # Parse query to get node filter and edge filter
            node_filter = query.get('node_filter')
            edge_filter = query.get('edge_filter')

            # Filter nodes based on the node_filter
            selected_nodes = self.filter_nodes(node_filter, edge_filter)
            print('Selected nodes:', selected_nodes)
            # Process each selected node to create time series
            for node in selected_nodes:
                self.process_element_for_time_series(node, ts_config, element_type='node')

        elif element_type == 'subgraph':
            # Process subgraphs based on subgraph_id or subgraph_label
            subgraph_id = query.get('subgraph_id')
            subgraph_label = query.get('subgraph_label')

            if subgraph_id:
                # Process a specific subgraph by ID
                subgraphs = [self.get_element('subgraph', subgraph_id)]
            elif subgraph_label:
                # Process all subgraphs with the given label
                subgraphs = [subgraph['data'] for subgraph in self.subgraphs.values() if subgraph['data'].label == subgraph_label]
            else:
                raise ValueError("Subgraph ID or label must be provided for subgraph time series generation.")

            for subgraph in subgraphs:
                self.process_element_for_time_series(subgraph, ts_config, element_type='subgraph')

        else:
            raise ValueError("Unsupported element type for time series generation: {}".format(element_type))

    def filter_nodes(self, node_filter, edge_filter):
        # This function now considers both node properties and edge connections
        filtered_nodes = []
        for node_id, node_data in self.graph.nodes(data=True):
            if node_filter(node_data):
                print('nodes filterrrrrrrr:', node_filter(node_data))
                # Check edges if an edge filter is specified
                if edge_filter:
                    connected_edges = self.graph.edges(node_id, data=True)
                    if any(edge_filter(edge) for _, _, edge in connected_edges):
                        filtered_nodes.append(node_data)
                else:
                    # No edge filter provided, add node based on node filter alone
                    filtered_nodes.append(node_data)
        return filtered_nodes

    def process_element_for_time_series(self, element, ts_config, element_type='node'):
        start_date = ts_config['start_date']
        end_date = ts_config.get('end_date', None)
        attribute = ts_config['attribute']
        aggregate_function = ts_config['aggregate_function']
        freq = ts_config.get('freq', 'D')

        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        values = []
        last_value = None

        for date in date_range:
            current_value = aggregate_function(self, element_type, element.oid, attribute, date)
            if current_value != last_value:
                values.append((date, current_value))
                last_value = current_value

        if values:
            timestamps, data_values = zip(*values)
            reshaped_data_values = np.array(data_values)[:, np.newaxis]
            tsid = self.id_generator.generate_timeseries_id()
            metadata = TimeSeriesMetadata(element.oid, '', element_type, attribute)
            time_series = TimeSeries(tsid, timestamps, [attribute], reshaped_data_values, metadata)
            self.time_series[tsid] = time_series
            print('Time series created:', tsid, time_series)
            self.add_property(element_type, element.oid, attribute, tsid)




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


    def display(self):
        print("Nodes:")
        for node_id, data in self.graph.nodes(data=True):
            print(f"Node {node_id}: {data}")

        print("\nEdges:")
        for source, target, key, data in self.graph.edges(keys=True, data=True):
            print(f"Edge {key} from {source} to {target}: {data['data']}")

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
        self.set_updated(False)  # Reset the flag after displaying
