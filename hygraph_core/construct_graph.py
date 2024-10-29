from hygraph_core.hygraph import HyGraph
from hygraph_core.timeseries_operators import TimeSeriesMetadata


def build_timeseries_similarity_graph(time_series_list, threshold, hygraph=None, similarity_metric='euclidean', edge_type='PGEdge', additional_option=None):
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

    # Step 1: Create TSNodes for each time series
    for idx, ts_data in enumerate(time_series_list):
        tsid = hygraph.id_generator.generate_timeseries_id()
        node_id = hygraph.id_generator.generate_node_id()

        # Create TimeSeries object
        timestamps = ts_data['timestamps']
        variables = ts_data['variables']
        data = ts_data['data']
        metadata = TimeSeriesMetadata(owner_id=node_id, element_type='TSNode')

        time_series = TimeSeries(tsid=tsid, timestamps=timestamps, variables=variables, data=data, metadata=metadata)
        hygraph.time_series[tsid] = time_series

        # Create TSNode
        ts_node = TSNode(oid=node_id, label='TSNode', time_series=time_series)
        hygraph.add_tsnode(ts_node)

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
                    # For PGEdge, store the similarity as a static property
                    properties = {'similarity': similarity}
                    hygraph.add_pgedge(
                        oid=edge_id,
                        source=ts_nodes[i].getId(),
                        target=ts_nodes[j].getId(),
                        label='SimilarityEdge',
                        start_time=timestamp,
                        properties=properties
                    )
                elif edge_type == 'TSEdge':
                    # For TSEdge, store the similarity over time as a TimeSeries
                    # Assuming we can compute similarity over time (e.g., sliding window)
                    similarity_ts = compute_similarity_timeseries(ts1, ts2, metric=similarity_metric)
                    hygraph.add_tsedge(
                        oid=edge_id,
                        source=ts_nodes[i].getId(),
                        target=ts_nodes[j].getId(),
                        label='SimilarityEdge',
                        time_series=similarity_ts
                    )
                else:
                    raise ValueError("edge_type must be 'PGEdge' or 'TSEdge'")

    return hygraph


