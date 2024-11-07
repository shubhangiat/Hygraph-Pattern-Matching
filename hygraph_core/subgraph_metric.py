# subgraph_metrics.py
import numpy as np

def count_edges_over_time(hygraph, element_type, oid, attribute, date):
    if element_type == 'subgraph':
        subgraph = hygraph.get_element('subgraph', oid)
        return sum(1 for _, _, edge_data in subgraph.edges(data=True) if edge_data['data'].start_time <= date)
    return 0

def count_nodes_over_time(hygraph, element_type, oid, attribute, date):
    if element_type == 'subgraph':
        subgraph = hygraph.get_element('subgraph', oid)
        return sum(1 for node_data in subgraph.nodes(data=True) if node_data['data'].start_time <= date)
    return 0

# Example aggregation of a time series attribute from nodes and edges in a subgraph
def aggregate_attribute_in_subgraph(hygraph, subgraph_id, element_type, attribute, date, aggregation='sum'):
    values = []

    # Determine whether to process nodes or edges
    if element_type == 'node':
        elements = hygraph.graph.nodes(data=True)
    elif element_type == 'edge':
        elements = hygraph.graph.edges(data=True)
    else:
        raise ValueError(f"Unsupported element_type: {element_type}. Use 'node' or 'edge'.")

    # Iterate over elements (nodes or edges) and collect attribute values
    for element_id, element_data in elements:
        data = element_data['data'] if element_type == 'node' else element_data
        if subgraph_id in data.membership:
            tsid = data.properties.get(attribute)
            if tsid and tsid in hygraph.time_series:
                ts_data = hygraph.time_series[tsid].data
                value = ts_data.sel(time=date, method='ffill').item()
                values.append(value)

    # Perform the desired aggregation
    if aggregation == 'sum':
        return np.sum(values)
    elif aggregation == 'avg':
        return np.mean(values)
    elif aggregation == 'min':
        return np.min(values)
    elif aggregation == 'max':
        return np.max(values)
    elif aggregation == 'count':
        return len(values)
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}. Supported values are 'sum', 'avg', 'min', 'max', 'count'.")


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
        node_filter = query.get('node_filter')
        edge_filter = query.get('edge_filter')

        selected_nodes = self.filter_nodes(node_filter, edge_filter)
        for node in selected_nodes:
            self.process_element_for_time_series(node, ts_config, element_type='node')

    elif element_type == 'subgraph':
        subgraph_id = query.get('subgraph_id')
        subgraph_label = query.get('subgraph_label')

        subgraphs = []
        if subgraph_id:
            subgraphs.append(self.get_element('subgraph', subgraph_id))
        elif subgraph_label:
            subgraphs.extend(subgraph['data'] for subgraph in self.subgraphs.values() if subgraph['data'].label == subgraph_label)

        for subgraph in subgraphs:
            self.process_element_for_time_series(subgraph, ts_config, element_type='subgraph')

    elif element_type == 'edge':
        edge_filter = query.get('edge_filter')
        direction = ts_config.get('direction', 'both')  # Handle direction for edges
        selected_edges = self.filter_edges(edge_filter, direction)
        for edge in selected_edges:
            self.process_element_for_time_series(edge, ts_config, element_type='edge')

    else:
        raise ValueError(f"Unsupported element type: {element_type}")
























import numpy as np
import pandas as pd
import xarray as xr

class TimeSeries:
    """
    Create and add a multivariate time series.
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
        """
        Append data to the time series, ensuring timestamps are unique and sequential.
        :param date: New timestamp for the data
        :param value: The value to append at this timestamp
        """
        # Convert the timestamp to nanosecond precision to match xarray's internal precision
        date = pd.Timestamp(date)
        if self.data.coords['time'].size > 0:
            last_timestamp = self.data.coords['time'][-1].item()
            if date <= last_timestamp:
                raise ValueError(f"New timestamp {date} must be after the last timestamp {last_timestamp}.")

        new_data = xr.DataArray([value], coords=[[date], self.data.coords['variable']], dims=['time', 'variable'])
        self.data = xr.concat([self.data, new_data], dim='time')

def dtw_distance(ts1: TimeSeries, ts2: TimeSeries, window: int = None, method: str = 'dependent') -> float:
    """
    Compute the DTW distance between two multivariate time series.

    :param ts1: First TimeSeries instance
    :param ts2: Second TimeSeries instance
    :param window: Maximum warping window size (int). If None, no constraint.
    :param method: 'dependent' or 'independent'. Determines how to handle multivariate data.
    :return: DTW distance (float)
    """
    if method == 'dependent':
        return dtw_dependent(ts1, ts2, window)
    elif method == 'independent':
        return dtw_independent(ts1, ts2, window)
    else:
        raise ValueError("Method must be 'dependent' or 'independent'.")

def dtw_dependent(ts1: TimeSeries, ts2: TimeSeries, window: int = None) -> float:
    """
    Compute Dependent DTW distance treating multivariate points as vectors.

    :param ts1: First TimeSeries instance
    :param ts2: Second TimeSeries instance
    :param window: Maximum warping window size (int). If None, no constraint.
    :return: DTW distance (float)
    """
    data1 = ts1.data.values  # Shape: (time, variable)
    data2 = ts2.data.values

    n, dim = data1.shape
    m, _ = data2.shape

    # Initialize cost matrix
    cost_matrix = np.full((n+1, m+1), np.inf)
    cost_matrix[0, 0] = 0

    # Set warping window
    if window is None:
        window = max(n, m)
    else:
        window = max(window, abs(n - m))

    # Compute cost matrix
    for i in range(1, n+1):
        for j in range(max(1, i - window), min(m+1, i + window + 1)):
            dist = np.linalg.norm(data1[i-1] - data2[j-1])
            cost = dist
            cost_matrix[i, j] = cost + min(
                cost_matrix[i-1, j],    # Insertion
                cost_matrix[i, j-1],    # Deletion
                cost_matrix[i-1, j-1]   # Match
            )

    dtw_distance = cost_matrix[n, m]
    return dtw_distance

def dtw_independent(ts1: TimeSeries, ts2: TimeSeries, window: int = None) -> float:
    """
    Compute Independent DTW distance by summing DTW distances over each variable.

    :param ts1: First TimeSeries instance
    :param ts2: Second TimeSeries instance
    :param window: Maximum warping window size (int). If None, no constraint.
    :return: Total DTW distance across all variables (float)
    """
    variables = ts1.data.coords['variable'].values
    total_distance = 0.0

    for var in variables:
        series1 = ts1.data.sel(variable=var).values
        series2 = ts2.data.sel(variable=var).values

        n = len(series1)
        m = len(series2)

        # Initialize cost matrix
        cost_matrix = np.full((n+1, m+1), np.inf)
        cost_matrix[0, 0] = 0

        # Set warping window
        if window is None:
            window_var = max(n, m)
        else:
            window_var = max(window, abs(n - m))

        # Compute cost matrix
        for i in range(1, n+1):
            for j in range(max(1, i - window_var), min(m+1, i + window_var + 1)):
                dist = abs(series1[i-1] - series2[j-1])
                cost = dist
                cost_matrix[i, j] = cost + min(
                    cost_matrix[i-1, j],    # Insertion
                    cost_matrix[i, j-1],    # Deletion
                    cost_matrix[i-1, j-1]   # Match
                )

        dtw_distance_var = cost_matrix[n, m]
        total_distance += dtw_distance_var

    return total_distance
































