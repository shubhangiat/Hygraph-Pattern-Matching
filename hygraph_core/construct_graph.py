from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.stats import pearsonr

from hygraph_core.graph_operators import TSNode
from hygraph_core.hygraph import HyGraph, HyGraphQuery
from hygraph_core.timeseries_operators import TimeSeriesMetadata, TimeSeries


def build_timeseries_similarity_graph(time_series_list, threshold, node_label,ts_attr_list, hygraph=None, similarity_metric='euclidean', edge_type='PGEdge', additional_option=None):
    """
    Build a HyGraph where each time series is a TSNode, and edges are added based on similarity.

    :param time_series_list: List of dictionaries containing time series data. Each dict should have:
        - 'timestamps': List of timestamps
        - 'variables': List of variable names (for multivariate time series)
        - 'data': 2D list or numpy array of data corresponding to variables
    :param threshold: Similarity threshold to add an edge between two nodes
    :param hygraph: An existing HyGraph instance or None to create a new one
    :param similarity_metric: The metric to compute similarity ('euclidean', 'dtw', 'correlation')
    :param edge_type: Type of edge to create ('PGEdge' or 'TSEdge')
    :param additional_option: Additional option for edge creation (e.g., 'distance_weighted')
    :return: A HyGraph instance with TSNodes and edges based on similarity
    """
    if hygraph is None:
        hygraph = HyGraph()

    ts_nodes = []
    tsid_to_nodeid = {}
    start_time=datetime.now()
    # Step 1: Create TSNodes for each time series
    for idx, ts_data in enumerate(time_series_list):
        tsid = hygraph.id_generator.generate_timeseries_id()
        node_id = hygraph.id_generator.generate_node_id()

        # Create TimeSeries object
        timestamps = ts_data['timestamps']
        variables = ts_data['variables']
        data = ts_data['data']
        metadata = TimeSeriesMetadata(owner_id=node_id, element_type='TSNode', attribute=ts_attr_list[idx])

        time_series = TimeSeries(tsid=tsid, timestamps=timestamps, variables=variables, data=data, metadata=metadata)
        hygraph.time_series[tsid] = time_series
        # Create TSNode
        ts_node=hygraph.add_tsnode(oid=node_id, label=node_label, time_series=time_series)

        ts_nodes.append(ts_node)
        tsid_to_nodeid[tsid] = node_id

    # Step 2: Compute pairwise similarities and add edges
    for i in range(len(ts_nodes)):
        for j in range(i + 1, len(ts_nodes)):
            ts1 = ts_nodes[i].series
            ts2 = ts_nodes[j].series

            similarity = compute_similarity(ts1, ts2, metric=similarity_metric)

            if similarity >= threshold:
                # Create an edge between ts_nodes[i] and ts_nodes[j]
                edge_id = hygraph.id_generator.generate_edge_id()
                timestamp = datetime.now()

                if edge_type == 'PGEdge':
                    start_time=ts1.first_timestamp()
                    # For PGEdge, store the similarity as a static property
                    properties = {'similarity': similarity}
                    hygraph.add_pgedge(edge_id,ts_nodes[i].getId(),ts_nodes[j].getId(),'Similar',start_time,properties=properties)
                elif edge_type == 'TSEdge':
                    # For TSEdge, store the similarity over time as a TimeSeries
                    # Assuming we can compute similarity over time (e.g., sliding window)
                    similarity_ts = compute_similarity_timeseries(ts1, ts2, metric=similarity_metric)
                    hygraph.add_tsedge(edge_id,ts_nodes[i].getId(),ts_nodes[j].getId(),'Similar',similarity_ts)

                else:
                    raise ValueError("edge_type must be 'PGEdge' or 'TSEdge'")

    return hygraph


#utility function

def compute_similarity(ts1, ts2, metric='euclidean'):
    """
    Compute similarity between two TimeSeries objects.

    :param ts1: First TimeSeries object
    :param ts2: Second TimeSeries object
    :param metric: Similarity metric ('euclidean', 'dtw', 'correlation')
    :return: Similarity value (the higher, the more similar)
    """
    similarity=0

    if metric == 'euclidean':
        similarity= ts2.euclidean_distance(ts1)
    elif metric == 'dtw':
        similarity=ts2.dynamic_time_warping(ts1)
    elif metric == 'correlation':
        similarity=ts2.correlation_coefficient(ts1)
    elif metric == 'manhattan':
        similarity=ts2.manhattan_distance(ts1)
    elif metric=='cosine':
        similarity=ts2.cosine_similarity (ts1)
    else:
        raise ValueError(f"Unsupported similarity metric: {metric}")

    return similarity


def compute_similarity_timeseries(ts1, ts2, metric='euclidean', window_size=5):
    """
    Compute similarity over time between two TimeSeries objects using a sliding window.

    :param ts1: First TimeSeries object
    :param ts2: Second TimeSeries object
    :param metric: Similarity metric ('euclidean', 'dtw', 'correlation')
    :param window_size: Size of the sliding window
    :return: TimeSeries object representing similarity over time
    """
    timestamps = []
    similarities = []

    data1 = ts1.data.values
    data2 = ts2.data.values
    times1 = ts1.data.coords['time'].values
    times2 = ts2.data.coords['time'].values

    # Ensure same timestamps
    common_times = np.intersect1d(times1, times2)
    idx1 = np.isin(times1, common_times)
    idx2 = np.isin(times2, common_times)
    data1 = data1[idx1]
    data2 = data2[idx2]

    for i in range(len(common_times) - window_size + 1):
        window_data1 = data1[i:i + window_size]
        window_data2 = data2[i:i + window_size]
        timestamp = common_times[i + window_size - 1]

        if metric == 'euclidean':
            dist = np.linalg.norm(window_data1 - window_data2)
            similarity = 1 / (1 + dist)
        elif metric == 'dtw':
            dist, _ = fastdtw(window_data1, window_data2)
            similarity = 1 / (1 + dist)
        elif metric == 'correlation':
            # Flatten if multivariate
            wd1 = window_data1.flatten()
            wd2 = window_data2.flatten()
            corr, _ = pearsonr(wd1, wd2)
            similarity = corr
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")

        timestamps.append(timestamp)
        similarities.append([similarity])

    variables = ['similarity']
    tsid = None  # Since this is a temporary TimeSeries, we can set tsid to None
    metadata = TimeSeriesMetadata(owner_id=None, element_type='TSEdge')
    similarity_ts = TimeSeries(tsid=tsid, timestamps=timestamps, variables=variables, data=similarities, metadata=metadata)

    return similarity_ts


if __name__ == '__main__':
    # Sample time series data for stocks
    # Generate sample time series data for 10 stocks
    time_series_list = []
    node_label_list = []

    # Common timestamps
    timestamps = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Base patterns
    base_pattern = np.sin(np.linspace(0, 2 * np.pi, 100))

    # Create 10 stocks with variations
    for i in range(10):
        # Slightly modify the base pattern for each stock
        noise = np.random.normal(0, 0.1, 100)
        shift = np.random.uniform(-0.5, 0.5)
        data = (base_pattern + shift + noise).reshape(-1, 1)
        ts_data = {
            'timestamps': timestamps,
            'variables': ['Price'],
            'data': data
        }
        time_series_list.append(ts_data)
        node_label_list.append(f"Stock_{chr(65 + i)}")  # Labels: Stock_A, Stock_B, ..., Stock_J

    # Build the HyGraph
    threshold = 0.95  # High similarity threshold
    hygraph = build_timeseries_similarity_graph(
        time_series_list,
        threshold=threshold,
        node_label='Stock',
        ts_attr_list=node_label_list,
        similarity_metric='correlation',  # Using correlation as the similarity metric
        edge_type='PGEdge'
    )
  # Print the nodes and edges
    # Print the nodes
    print("Nodes:")
    for node_id, data in hygraph.graph.nodes(data=True):
        node = data.get('data')
        ts=hygraph.get_timeseries(node.series.tsid)
        print(f"Node ID: {node_id}, Label: {node.label}, TSID: {node.series.tsid}, Attribute: {ts.metadata.attribute}")

    # Print the edges
    print("\nEdges:")
    for u, v, key, data in hygraph.graph.edges(keys=True, data=True):
        edge = data.get('data')
        similarity = edge.get_static_property('similarity')
        print(f"Edge ID: {edge.oid}, From: {u}, To: {v}, Similarity: {similarity:.4f}")

    # Apply community detection (e.g., Girvan-Newman algorithm)
    communities = nx.algorithms.community.girvan_newman(hygraph.graph)
    # Get the first level of communities
    top_level_communities = next(communities)
    community_list = [list(c) for c in top_level_communities]

    # Print the clusters
    for idx, community in enumerate(community_list):
        stock_names = [hygraph.graph.nodes[node]['data'].label for node in community]
        print(f"Community {idx + 1}: {stock_names}")

    # Calculate centrality measures
    centrality = nx.degree_centrality(hygraph.graph)

    # Get the top N central stocks
    top_central_stocks = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]

    # Print the most central stocks
    for node_id, centrality_value in top_central_stocks:
        stock = hygraph.graph.nodes[node_id]['data']
        print(f"Stock: {stock.label}, Centrality: {centrality_value:.4f}")

    # Retrieve the historical in-degree of a node
    query = HyGraphQuery(hygraph)

    stockA = hygraph.get_nodes_by_label('')
    ts_in_degree = node_degree_history = hygraph.get_node_degree_over_time(node_id=1, degree_type='in',
                                                                           return_type='history')
    node_degree_history.display_time_series()