import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from fastdtw import fastdtw
from numpy.lib.utils import source
from scipy.spatial.distance import euclidean
from datetime import datetime, timedelta
from igraph import Graph as IGraph

from hygraph_core.graph_operators import Edge, TSNode, PGNode, PGEdge, TSEdge
from hygraph_core.timeseries_operators import TimeSeries, TimeSeriesMetadata
from idGenerator import IDGenerator
from oberserver import Subject
FAR_FUTURE_DATE = datetime(2100, 12, 31, 23, 59, 59)
from constraints import parse_datetime



class HyGraph:
    def __init__(self):
        self.graph = nx.MultiGraph()
        self.time_series = {}
        self.subgraphs = {}
        self.id_generator = IDGenerator()
        self.updated = False  # Flag to track updates
        self.query =None

    def add_pg_node(self, oid, label, start_time, end_time=None, properties=None):
        """
        Add a new PGNode to the graph with specified properties.

        :param oid: Object ID of the node.
        :param label: Label for the node.
        :param start_time: Start time for the node.
        :param end_time: End time for the node (optional).
        :param properties: A dictionary of properties to assign to the node.
        """
        # Create a new PGNode instance
        pg_node = PGNode(oid, label, start_time, end_time)

        # Add any specified properties to the node
        if properties:
            pg_node.properties.update(properties)

        # Add the node to the graph
        self.add_node(pg_node)

        print(f"PGNode {oid} added with label '{label}' and properties {pg_node.properties}")

        return pg_node

    def add_ts_node(self, oid, label, time_series):
        """
        Add a new TSNode (Time Series Node) to the graph with a time series and label.

        :param oid: Object ID of the node.
        :param label: Label for the node.
        :param time_series: Time series data to attach to the node.
        """
        # Create a new TSNode instance
        ts_node = TSNode(oid, label, time_series)

        # Add the node to the graph
        self.add_node(ts_node)

        print(f"TSNode {oid} added with label '{label}' and time series data {time_series}")

        return ts_node

    def remove_node(self, oid):
        """
        Delete a node from the graph.

        :param oid: The Object ID of the node to be deleted.
        :raises ValueError: If the node does not exist in the graph.
        """
        if oid not in self.graph.nodes:
            raise ValueError(f"Node with ID {oid} does not exist.")

        # Remove the node from the graph
        self.graph.remove_node(oid)

        # Notify that the graph has been updated
        self.set_updated()

        print(f"Node {oid} has been removed from the graph.")

    def update_node_properties(self, oid, properties):
        """
        Update the properties of an existing node in the graph.

        :param oid: The Object ID of the node to be updated.
        :param properties: A dictionary containing the new or updated properties for the node.
        :raises ValueError: If the node does not exist in the graph.
        """
        if oid not in self.graph.nodes:
            raise ValueError(f"Node with ID {oid} does not exist.")

        # Get the node from the graph
        node = self.graph.nodes[oid]['data']

        # Update the node's properties with the new values
        node.properties.update(properties)

        # Notify that the graph has been updated
        self.set_updated()

        print(f"Node {oid} properties updated: {properties}")

        # Notify observers if needed
        node.notify()

    def get_node_property(self, oid, property_key):
        """
        Retrieve a specific property from a node.

        :param oid: Object ID of the node.
        :param property_key: The key of the property to retrieve.
        :return: The value of the property, or None if the property does not exist.
        :raises ValueError: If the node does not exist in the graph.
        """
        if oid not in self.graph.nodes:
            raise ValueError(f"Node with ID {oid} does not exist.")

        # Get the node from the graph
        node = self.graph.nodes[oid]['data']

        # Retrieve the property if it exists
        return node.properties.get(property_key, None)

    def set_node_property(self, oid, property_key, value):
        """
        Set or update a specific property of a node.

        :param oid: Object ID of the node.
        :param property_key: The key of the property to set or update.
        :param value: The value to assign to the property.
        :raises ValueError: If the node does not exist in the graph.
        """
        if oid not in self.graph.nodes:
            raise ValueError(f"Node with ID {oid} does not exist.")

        # Get the node from the graph
        node = self.graph.nodes[oid]['data']

        # Set or update the property
        node.properties[property_key] = value

        # Notify that the graph has been updated
        self.set_updated()
        node.notify()

        print(f"Property '{property_key}' set/updated to '{value}' for Node {oid}.")

    def add_pg_edge(self, oid, source_oid, target_oid, label, start_time, end_time=None, properties=None):
        """
        Add a new PGEdge to the graph with specified properties.

        :param oid: Object ID of the edge.
        :param source_oid: The Object ID of the source node.
        :param target_oid: The Object ID of the target node.
        :param label: The label of the edge.
        :param start_time: The start time of the edge.
        :param end_time: The end time of the edge (optional).
        :param properties: A dictionary of additional properties for the edge (optional).
        :raises ValueError: If the source or target node does not exist in the graph.
        """
        # Ensure the source and target nodes exist
        if source_oid not in self.graph.nodes:
            raise ValueError(f"Source node with ID {source_oid} does not exist.")
        if target_oid not in self.graph.nodes:
            raise ValueError(f"Target node with ID {target_oid} does not exist.")

        if not properties:
            properties = {}

        # Create a new PGEdge instance
        pg_edge = PGEdge(oid, source_oid, target_oid, label, properties, start_time, end_time)

        # Add the edge to the graph
        self.graph.add_edge(source_oid, target_oid, key=pg_edge.oid, data=pg_edge)

        # Mark the graph as updated
        self.set_updated()

        print(
            f"PGEdge {oid} added between Node {source_oid} and Node {target_oid} with label '{label}' and properties {pg_edge.properties}")

        return pg_edge

    def add_ts_edge(self, oid, source_oid, target_oid, label, time_series, start_time, end_time=None):
        """
        Add a new TSEdge to the graph with a time series and label.

        :param oid: Object ID of the edge.
        :param source_oid: The Object ID of the source node.
        :param target_oid: The Object ID of the target node.
        :param label: The label of the edge.
        :param time_series: The time series data to attach to the edge.
        :raises ValueError: If the source or target node does not exist in the graph.
        """
        # Ensure the source and target nodes exist
        if source_oid not in self.graph.nodes:
            raise ValueError(f"Source node with ID {source_oid} does not exist.")
        if target_oid not in self.graph.nodes:
            raise ValueError(f"Target node with ID {target_oid} does not exist.")

        # Create a new TSEdge instance
        ts_edge = TSEdge(oid, source_oid, target_oid, label, start_time, time_series)

        # Add the edge to the graph
        self.graph.add_edge(source_oid, target_oid, key=ts_edge.oid, data=ts_edge)

        # Mark the graph as updated
        self.set_updated()

        print(f"TSEdge {oid} added between Node {source_oid} and Node {target_oid} with label '{label}' and time series data.")

        return ts_edge

    def remove_edge(self, oid):
        """
        Remove an edge from the graph.

        :param oid: The Object ID of the edge to be deleted.
        :raises ValueError: If the edge does not exist in the graph.
        """
        # Find the edge by OID
        edge = self.get_edge(oid)

        # Mark the graph as updated
        self.set_updated()

        print(f"Edge {oid} has been removed from the graph.")

    def update_edge_properties(self, oid, properties):
        """
        Update the properties of an existing edge in the graph.

        :param oid: Object ID of the edge.
        :param properties: A dictionary containing the new or updated properties for the edge.
        :raises ValueError: If the edge does not exist in the graph.
        """
        # Find the edge by OID
        edge = self.get_edge(oid)

        # Update the edge's properties with the new values
        edge["data"].properties.update(properties)

        # Mark the graph as updated
        self.set_updated()

        # Notify observers
        edge["data"].notify()

        print(f"Edge {oid} properties updated: {properties}")


    def get_edge_property(self, oid, property_key):
        """
        Retrieve a specific property from an edge.

        :param oid: Object ID of the edge.
        :param property_key: The key of the property to retrieve.
        :return: The value of the property, or None if the property does not exist.
        :raises ValueError: If the edge does not exist in the graph.
        """
        # Find the edge by OID
        edge = self.get_edge(oid)

        # Retrieve the property if it exists
        return edge["data"].properties.get(property_key, None)

    def set_edge_property(self, oid, property_key, value):
        """
        Set or update a specific property of an edge.

        :param oid: Object ID of the edge.
        :param property_key: The key of the property to set or update.
        :param value: The value to assign to the property.
        :raises ValueError: If the edge does not exist in the graph.
        """
        # Find the edge by OID
        edge = self.get_edge(oid)

        # Set or update the property in the edge's properties dictionary
        edge['data'].properties[property_key] = value

        # Mark the graph as updated
        self.set_updated()
        edge['data'].notify()

        print(f"Property '{property_key}' set to '{value}' for Edge {oid}.")

    def set_updated(self, value=True):
        self.updated = value

    def add_node(self, node):
        self.graph.add_node(node.oid, data=node)
        print("Adding node with:", node.label, node.oid)
        self.set_updated()

    def add_edge(self, edge):
        self.graph.add_edge(edge.source, edge.target, key=edge.oid, data=edge)
        print("Adding edge with:", edge.label, edge.start_time, edge.oid)
        self.set_updated()

    def add_subgraph(self, subgraph):
        print(f"Adding subgraph with ID: {subgraph.subgraph_id}")
        subgraph_view = nx.subgraph_view(self.graph, filter_node=subgraph.filter_func, filter_edge=subgraph.filter_func)
        self.subgraphs[subgraph.subgraph_id] = {'view': subgraph_view, 'data': subgraph}

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
        self.query = query
        self.batch_process()
        print(f"Query added: {query}")
        # If needed, you can reinitialize the observer or notify it about the new query

    def set_query(self, query):
        """
        Sets or updates the query for the HyGraph instance.
        This query will be used by the observer to generate time series.
        """
        self.query = query
        print("Query has been set/updated.")

    def get_query(self):
        """
        Returns the current query  being used by the HyGraph instance.
        If an index is provided, return the specific query at that index.
        """
        return self.query

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

    def add_membership(self, element_id, timestamp, subgraph_ids,element_type):
        """
           Adds subgraph memberships to the element's TimeSeries at the given timestamp.
           """
        element = self.get_element(element_type, element_id)
        if not element:
            print(f"No element found with ID {element_id}")
            return

        if element.membership is None:
            tsid = self.id_generator.generate_timeseries_id()
            membership_string = ' '.join(subgraph_ids)
            print('membership', membership_string)
            metadata=TimeSeriesMetadata(element_id,'membership',element_type)
            time_series = TimeSeries(tsid, [timestamp], ['membership'], membership_string,metadata)
            self.time_series[tsid] = time_series
            element.membership = tsid
            print(f"Timeseries created for: {element_type} {element_id}: {time_series}")
        else:

            tsid = element.membership
            time_series = self.time_series[tsid]
            # Initialize the membership string for each update
            # Ensure there is at least one entry to extract the last state
            # Get the last known membership state
            if time_series.data.shape[0] > 0:
                last_entry = time_series.data.isel(time=-1).item().split()
            else:
                last_entry = []
            # Combine, deduplicate, and sort memberships
            updated_membership = sorted(set(last_entry + subgraph_ids))
            membership_string = ' '.join(updated_membership)
            print('membership', membership_string)
            time_series.append_data(timestamp, membership_string)
            print(f"Updated membership for {element_type} {element_id} at {timestamp}: add {subgraph_ids}")

    def remove_membership(self, element_id, timestamp, subgraph_ids,element_type):
        """
        Removes subgraph memberships from the element's TimeSeries at the given timestamp.
        """
        element = self.get_element(element_type, element_id)
        if not element:
            print(f"No element found with ID {element_id}")
            return

        if element.membership is None:
            print(f"No TimeSeries initialized for {element_type} {element_id}")

        else:
            tsid = element.membership
            time_series = self.time_series[tsid]
            # Get the last known membership state
            if time_series.data.shape[0] > 0:
                last_entry = time_series.data.isel(time=-1).item().split()
            else:
                last_entry = []
            # Remove specified subgraph IDs
            updated_membership = [sg for sg in last_entry if sg not in subgraph_ids]
            membership_string = ' '.join(updated_membership)

            time_series.append_data(timestamp, membership_string)
            print(f"Updated membership for {element_type} {element_id} at {timestamp}: remove {subgraph_ids}")

    def update_membership(self, element, timestamps, subgraph_ids,action):
        # Ensure that timestamps and subgraph_ids are aligned
        if len(timestamps) != len(subgraph_ids):
            raise ValueError("Timestamps and subgraph IDs arrays must have the same length.")

        tsid = element.membership.get('subgraph_membership') or self.id_generator.generate_timeseries_id()

        if tsid not in self.time_series:

            membership_series = TimeSeries(tsid, timestamps, ['membership'], [subgraph_ids])
            self.time_series[tsid] = membership_series
        else:
            membership_series = self.time_series[tsid]
            # Update existing time series based on action
            for timestamp, subgraph_id in zip(timestamps, subgraph_ids):
                if action == 'add':
                    # Use the existing append_data method to add the subgraph_id
                    membership_series.append_data(timestamp, subgraph_id)
                elif action == 'remove':
                    # Remove the subgraph ID for the given timestamp
                    index = membership_series.timestamps.index(
                        timestamp) if timestamp in membership_series.timestamps else None
                    if index is not None:
                        existing_data = membership_series.data[index]
                        if subgraph_id in existing_data:
                            existing_data.remove(subgraph_id)  # Remove subgraph ID if present
                            # Optionally: If you want to mark it as "removed" without deleting
                            if not existing_data:
                                existing_data.append(None)  # Placeholder to indicate removal

            element.membership['subgraph_membership'] = tsid
            print(
                f"Added subgraph membership to {type(element).__name__.lower()} {element.oid} with time series ID {tsid}.")

    def get_node(self, oid):
        if oid not in self.graph.nodes:
            raise ValueError(f"Node with ID {oid} does not exist.")
        return self.graph.nodes[oid]

    def get_edge(self, oid):
        """
        Retrieve an edge from the graph using its Object ID (oid).
        """
        # Iterate through all edges in the graph
        for u, v, key, edge_data in self.graph.edges(data=True, keys=True):
            # If the edge's Object ID matches the requested oid, return the edge
            if key == oid:
                return edge_data

        # Raise an error if the edge with the given oid is not found
        raise ValueError(f"Edge with ID {oid} does not exist.")


    def get_time_series(self, tsid):
        if tsid not in self.time_series:
            raise ValueError(f"Time series with ID {tsid} does not exist.")
        return self.time_series[tsid].data

    def get_subgraph(self, subgraph_id):
        if subgraph_id not in self.subgraphs:
            raise ValueError(f"Subgraph with ID {subgraph_id} does not exist.")
        return self.subgraphs[subgraph_id]

    def add_time_series(self, timestamps, variables, data, metadata=None):
        """
        Add a new time series to the hygraph.

        :param timestamps: List of timestamps for the time series
        :param variables: List of variables for the time series
        :param data: 2D array-like structure with data corresponding to timestamps and variables
        :param metadata: Optional metadata for the time series
        :return: The ID of the newly created time series
        """
        # Generate a new time series ID
        time_series_id = self.id_generator.generate_timeseries_id()

        # Create the time series
        new_time_series = TimeSeries(time_series_id, timestamps, variables, data, metadata)

        # Store the time series in the hygraph
        self.time_series[time_series_id] = new_time_series

        print(f"New time series added with ID {time_series_id}")
        return time_series_id
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
            # Process each selected node to create time series
            for node_data in selected_nodes:

                data = node_data['data']
                attribute = self.query['time_series_config']['attribute']
                tsid = data.properties.get(attribute)
                if tsid and tsid in self.time_series:
                    print(f"Appending to existing time series {tsid} for node {data.oid}")
                    current_value = self.query['time_series_config']['aggregate_function'](
                        self, 'node', data.oid, attribute, datetime.now()
                    )
                    self.append_time_series(tsid, datetime.now(), current_value)
                else:
                    print(f"Creating new time series for node {data.oid}")
                    self.process_element_for_time_series(node_data, ts_config, element_type='node')

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
        # This function now considers both node properties.py and edge connections
        filtered_nodes = []
        for node_id, node_data in self.graph.nodes(data=True):
            if node_filter(node_data):
                # Check edges if an edge filter is specified
                if edge_filter:
                    connected_edges = self.graph.edges(node_id, data=True)
                    if any(edge_filter(edge) for _, _, edge in connected_edges):
                        filtered_nodes.append(node_data)
                else:
                    # No edge filter provided, add node based on node filter alone
                    filtered_nodes.append(node_data)
        return filtered_nodes

    def process_element_for_time_series(self, element_data, ts_config, element_type='node'):
        element = element_data['data']
        #  start_date = ts_config['start_date']
        #end_date = ts_config.get('end_date', None)
        attribute = ts_config['attribute']
        aggregate_function = ts_config['aggregate_function']
        #freq = ts_config.get('freq', 'D')
        element_start_time = parse_datetime(element.start_time) if isinstance(element.start_time, str) else element.start_time
        element_end_time = parse_datetime(element.end_time) if isinstance(element.end_time, str) else element.end_time
        # Check if the element is within the time range
        #if pd.isna(end_date):
        #   end_date =datetime.now()
        #if not (element_start_time <= end_date and element_end_time >= start_date):
        #   print(f"Skipping {element_type} {element.oid} as it is outside the query time range.")
        #   return
        #date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        values = []
        last_value = None
        # for date in date_range:
        #   current_value = aggregate_function(self, element_type, element.oid, attribute, date)
        #   values.append((date, current_value))
        if values:
            timestamps, data_values = zip(*values)
            reshaped_data_values = np.array(data_values)[:, np.newaxis]
            tsid = self.id_generator.generate_timeseries_id()
            metadata = TimeSeriesMetadata(element.oid, element_type, attribute)
            time_series = TimeSeries(tsid, timestamps, [attribute], reshaped_data_values, metadata)

            self.time_series[tsid] = time_series
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

    def batch_process(self):
        """Batch processing function to process all nodes and edges."""
        print("Starting batch processing...")
        self.create_time_series_from_graph(self.query)  # Update time series for all nodes and edges
        print("Batch processing completed.")

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

        print("\nTime Series:", len(self.time_series))
        for tsid, ts in self.time_series.items():
            print(f"Time Series {tsid}: {ts.metadata.owner_id}")
            variables = [str(var) for var in ts.data.coords['variable'].values]
            print(f"Variables: {', '.join(variables)}")
            ts_df = ts.data.to_dataframe('value').reset_index()
            grouped = ts_df.groupby('time')
            for time, group in grouped:
                values = [f" {row['value']}" for idx, row in group.iterrows()]
                row_str = ", ".join(values)
                print(f"{time}, {row_str}")
        self.set_updated(False)  # Reset the flag after displaying

#utility functions
