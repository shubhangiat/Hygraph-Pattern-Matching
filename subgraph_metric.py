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
