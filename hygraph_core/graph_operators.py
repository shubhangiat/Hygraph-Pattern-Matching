from datetime import datetime

import numpy as np

from hygraph_core.hygraph import HyGraph
from hygraph_core.oberserver import Subject
from hygraph_core.timeseries_operators import TimeSeries


class Node(Subject):
    def __init__(self, oid, label, node_id=None):
        super().__init__()  # Initialize the Subject
        self.oid = oid
        self.node_id = node_id  # External ID from CSV file
        self.label = label
        self.membership = None
    def __repr__(self):
        return f"Node(oid={self.oid}, label={self.label}, membership={self.membership})"
    def get_type(self):
        """
        Returns the type of node. Should be overridden by subclasses.
        """
        return "Node"
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
    def get_type(self):
        return "PGNode"
class TSNode(Node):
    def __init__(self, oid, label,time_series):
        super().__init__(oid, label)
        self.series = time_series
    def __repr__(self):
        base_str = super().__repr__()
        return f"{base_str}, series={self.series}"

    def get_type(self):
        return "TSNode"
class Edge(Subject):
    def __init__(self, oid, source, target, label, start_time, end_time=None, edge_id=None):
        super().__init__()
        self.oid = oid
        self.source = source
        self.target = target
        self.label = label
        self.edge_id = edge_id  # External ID from CSV file
        self.properties = {}
        self.membership = None
    def __repr__(self):
        properties_str = ', '.join(f"{k}: {v}" for k, v in self.properties.items())
        return f"Edge(oid={self.oid}, source={self.source}, target={self.target}, label={self.label}, membership={self.membership} properties={{ {properties_str} }})"

class PGEdge(Edge):
    def __init__(self, oid, label, start_time, end_time=None, node_id=None):
        super().__init__(oid, label,node_id)
        self.start_time = start_time
        self.end_time = end_time
        self.properties = {}
    def __repr__(self):
        properties_str = ', '.join(f"{k}: {v}" for k, v in self.properties.items())
        base_str = super().__repr__()
        return f"{base_str}, start_time={self.start_time}, end_time={self.end_time}, properties={{ {properties_str} }}"

class TSEdge(Edge):
    def __init__(self, oid, label,time_series):
        super().__init__(oid, label)
        self.series = time_series
    def __repr__(self):
        base_str = super().__repr__()
        return f"{base_str}, series={self.series}"
class Subgraph(Subject):
    def __init__(self, subgraph_id, label, start_time, end_time=None, filter_func=None):
        super().__init__()
        self.subgraph_id = subgraph_id
        self.label = label
        self.start_time = start_time
        self.end_time = end_time
        self.properties = {}
        self.filter_func = filter_func

    def __repr__(self):
        properties_str = ', '.join(f"{k}: {v}" for k, v in self.properties.items())
        return f"Subgraph(id={self.subgraph_id}, label={self.label}, start_time={self.start_time}, end_time={self.end_time}, properties={{ {properties_str} }})"


class StaticProperty:
    def __init__(self, name, value):
        self.last_updated = None
        self.name = name
        self.value = value

    def get_value(self):
        return self.value

    def set_value(self, new_value):
        self.value = new_value
        self.last_updated = datetime.now()
class TemporalProperty:
    def __init__(self, name, time_series_id,hygraph):
        self.name = name
        self.hygraph=hygraph
        #assert isinstance(time_series, TimeSeries), "time_series must be of type TimeSeries"
        self.time_series_id = time_series_id  # Instance of TimeSeries class

    def get_time_series(self) -> TimeSeries:
        return self.hygraph.time_series.get(self.time_series_id)

    def get_value_at(self, timestamp):
        return self.get_time_series().get_value_at_timestamp(timestamp)

    def apply_aggregation(self, function_name,start_time=None,end_time=None):
        """
            Apply an aggregation function (like 'sum', 'mean', etc.) to the time series data,
            optionally filtered by a time range (start_time to end_time).

            :param function_name: Aggregation function ('sum', 'mean', 'min', 'max', 'count', etc.)
            :param start_time: Start of the time range for filtering (optional).
            :param end_time: End of the time range for filtering (optional).
            :return: Aggregated value based on the specified function.
            """
        filtered_data= self.get_time_series().data
        if start_time and end_time:
            # Filter data within the given time range
            filtered_data = self.get_time_series().data.sel(time=slice(start_time, end_time))
        elif start_time and not end_time:
            # Filter data starting from start_time to the last available timestamp
            filtered_data = self.get_time_series().data.sel(time=slice(start_time, None))
        elif not start_time and end_time:
            # Filter data from the first available timestamp to end_time
            filtered_data = self.get_time_series().data.sel(time=slice(None, end_time))

        return filtered_data.apply_aggregation(function_name)
    def get_timestamp(self,value):
        return self.get_time_series().get_timestamp_at_value(value)

    def get_last_value(self):
        return self.get_time_series().last_value()

    def get_first_value(self):
        return self.get_time_series().first_value()

class GraphElement:
    """
    Superclass for all graph elements (node, edge, subgraph) with shared property management.
    """
    def __init__(self,oid, label,membership,hygraph):
        self.oid = oid
        self.membership=membership
        self.label=label
        self.hygraph=hygraph
        self.static_properties = {}
        self.temporal_properties = {}

    def get_property_type(self, property_name):
        """
        Determine the type of the property (static or dynamic).
        """
        if property_name in self.static_properties:
            return 'static'
        elif property_name in self.temporal_properties:
            return 'dynamic'
        else:
            return None
    def add_static_property(self, name, value):
        self.static_properties[name] = StaticProperty(name, value)

    def add_temporal_property(self, name, timestamps, variables, data):
        """
        Add a new temporal property to the graph element.

        :param name: Name of the temporal property
        :param timestamps: List of timestamps for the time series
        :param variables: List of variables for the time series
        :param data: 2D array-like structure with data corresponding to timestamps and variables
        """
        if name in self.temporal_properties:
            raise ValueError(f"Temporal property '{name}' already exists.")

        # Create the time series in hygraph and get the time series ID
        time_series_id = self.hygraph.add_time_series(timestamps, variables, data)

        # Create a TemporalProperty and add it to the element's temporal_properties
        temporal_property = TemporalProperty(name, time_series_id, self.hygraph)
        self.temporal_properties[name] = temporal_property

        print(f"Temporal property '{name}' added with time series ID {time_series_id}")
    def get_static_property(self, name):
        if name in self.static_properties:
            return self.static_properties[name]
        else:
            raise ValueError(f"Static property {name} not found.")

    def get_temporal_property(self, name, limit=None, order='first'):
        """
        Retrieve the specified temporal property (dynamic property) with an option to limit the number of data points.

        Parameters:
        - name (str): The name of the property to retrieve.
        - limit (int, optional): The maximum number of data points to retrieve.
        - order (str, optional): Whether to retrieve the 'first' or 'last' N data points. Defaults to 'first'.
        """
        if name in self.temporal_properties:
            timeseries = self.temporal_properties[name]

            # Call the display method to show time series data
            timeseries.display(limit=limit, order=order)

            return timeseries
        else:
            raise ValueError(f"Dynamic property {name} not found.")

    def get_all_properties(self, display_ts=False,limit=None,order='first'):
        """
            Retrieve and display both static and temporal properties of the graph element.

            Parameters:
            - limit (int, optional): The maximum number of data points to retrieve for temporal properties.
            - order (str, optional): Whether to retrieve the 'first' or 'last' N data points for temporal properties. Defaults to 'first'.
            """
        print("Static Properties:")
        for name, prop in self.static_properties.items():
            print(f"{name}: {prop.get_value()} (Last updated: {prop.timestamp})")

        print("\nTemporal Properties:")
        for name, ts in self.temporal_properties.items():
            if display_ts:
                ts.display_time_series(limit,order)
            else:
                print(f"{name}: TimeSeries ID: {ts.tsid}")

    def get_property(self, property_name,limit=None, order='last'):
        """
        Retrieve either static or temporal property.
        """
        if property_name in self.static_properties:
            return self.get_static_property(property_name)
        elif property_name in self.temporal_properties:
            return self.get_temporal_property(property_name)
        else:
            raise ValueError(f"Property {property_name} not found.")

    def _get_static_properties(self):
        """
        Retrieve all static properties from the element.
        """
        return {key: value for key, value in self.static_properties.items()}

    def _get_temporal_properties(self):
        """
        Retrieve all temporal (dynamic) properties (time-series) from the element.
        """
        return {key: value for key, value in self.temporal_properties.items()}


    def getId(self):
        """
        Get the ID of the graph element.
        """
        return self.oid

    def getLabel(self):
        """
        Get the label of the graph element.
        """
        return self.label
    def get_membership(self):
        """
        Get the membership time series of the element.
        """
        if self.membership:
            tsid = self.membership  # Membership is a time series ID
            if tsid in self.hygraph.time_series:
                return self.hygraph.time_series[tsid]
            else:
                return None
        return None









