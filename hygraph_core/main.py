from datetime import datetime, timedelta
import time

import numpy as np
import pandas as pd
from watchdog.observers import Observer

from HyGraphFileLoaderBatch import HyGraphBatchProcessor
from hygraph import HyGraph, Edge, PGNode, HyGraphQuery
from fileProcessing import NodeFileHandler, EdgeFileHandler, HyGraphFileLoader
import os

from hygraph_core.timeseries_operators import TimeSeries, TimeSeriesMetadata

base_dir = os.path.dirname(os.path.abspath(__file__))

# Return number of edges/ neighbors
def connection_count_aggregate_function(graph, element_type, oid, attribute, date):
    if element_type == 'node':
        print("number of connections ", len(list(graph.graph.neighbors(oid))))
        return len(list(graph.graph.neighbors(oid)))
    return 0

# Return number of edges in the graph created until given timestamp
def count_edges_in_subgraph(subgraph, date):
    return sum(1 for _, _, edge_data in subgraph.view.edges(data=True) if edge_data['data'].start_time <= date)


if __name__ == "__main__":
    hygraph = HyGraph()  # Initialize an empty HyGraph instance

    #Add mock PGNode stations with static properties including 'capacity'
    node1 = hygraph.add_pgnode(
                oid=1, 
                label='Station', 
                start_time=datetime.now() - timedelta(hours=7),
                properties={'capacity': 100, 'name': 'Station A'}
            )
    node2 = hygraph.add_pgnode(
                oid=2,
                label='Station', 
                start_time=datetime.now() - timedelta(hours=8),
                properties={'capacity': 40, 'name': 'Station B'}
            )
    node3 = hygraph.add_pgnode(
                oid=3,
                label='Station',
                start_time=datetime.now() - timedelta(hours=7),
                properties={'capacity': 60, 'name': 'Station C'}
            )

    try:
        data = hygraph.graph.nodes[1]
        print("Data keys for node:", data.keys())
        print("Specific data content:", {k: data[k] for k in data.keys()})
    except Exception as e:
        print(f"Error accessing data for node {1}: {e}")

    edge1 = hygraph.add_pgedge(
                oid=4,
                source=1,
                target=2,
                label='Trip',
                start_time = datetime.now() - timedelta(hours=4)
            )
    edge2 = hygraph.add_pgedge(
                oid=5,
                source=1,
                target=3,
                label='Trip',
                start_time=datetime.now() - timedelta(hours=3)
            )
    
    # Create a TimeSeries object
    timestamps = ['2023-01-01', '2023-01-02', '2023-01-03']
    data = [[10], [20], [15]]  # Values associated with the timestamps
    variables = ['BikeAvailability']

    time_series = hygraph.add_time_series(timestamps, variables, data)

    time_series.display_time_series()
    node1.add_temporal_property("bikeavailable", time_series, hygraph)
    node1.add_static_property("lat", 20, hygraph)
    edge1.add_static_property("bike_type","electric",hygraph)
    print('Node with label station: ', hygraph.get_nodes_by_label('Station'))

    def condition(ts):
        return ts.sum() > 40

    print("here is the first condition ", hygraph.get_nodes_by_temporal_property("bikeavailable", condition))


    def condition_static(static_prop):
        return static_prop.get_value() == 60


    print("here is the second condition ", hygraph.get_nodes_by_static_property("capacity", condition_static))

    query = HyGraphQuery(hygraph)

    def condition_func(node):
        print("Node structure:", node.get('properties', {}).get('capacity').get_value()> 50)  # This will show you what `node` contains
        return node.get('properties', {}).get('capacity').get_value()> 50

    results = (
        query
        .match_node(alias='station', node_id=1)  # Match edge by key only
        .return_(
            name=lambda n: n['station'].get_static_property('name'),
        )
        .execute()
    )
    print("the edge",edge1)
    for result in results:
        print(result)

    print('actua one : ', len(results))
    for node in results:
        print(node)
    # Retrieve the historical in-degree of a node
    ts_in_degree=node_degree_history = hygraph.get_node_degree_over_time(node_id=1, degree_type='in', return_type='history')
    node_degree_history.display_time_series()

    # Retrieve the current out-degree of a node
    current_out_degree = hygraph.get_node_degree_over_time(node_id=1, degree_type='out', return_type='current')
    print(f"Current Out-Degree: {current_out_degree}")
    ts_out_degree=hygraph.get_node_degree_over_time(node_id=1,degree_type='out',return_type='history')

    both_degree= ts_in_degree.aggregate_time_series_cumulative(ts_out_degree,'both_degree').display_time_series()

    hygraph.add_pgedge(oid=60,source=2,target=1,label='Trip',start_time=datetime.now() + timedelta(hours=5))
    ts_in_degree = node_degree_history = hygraph.get_node_degree_over_time(node_id=1, degree_type='in',return_type='history')
    
    ts_out_degree = hygraph.get_node_degree_over_time(node_id=1, degree_type='out', return_type='history')
    ts_in_degree.aggregate_time_series_cumulative(ts_out_degree,'both_degree').display_time_series()


    def create_trip_series(start_time, length=10):
        timestamps = [start_time + timedelta(minutes=5 * i) for i in range(length)]
        trip_counts = np.random.randint(5, 20, size=length)
        data = trip_counts.reshape((length, 1))  # Reshape to (length, 1)
        metadata = TimeSeriesMetadata(owner_id=None)

        return hygraph.add_time_series(
            timestamps=timestamps,
            variables=['trip_count'],
            data=data,
            metadata=metadata
        )


    fixed_date = datetime(2023, 1, 1, 0, 0, 0)
    ts3= create_trip_series(start_time=fixed_date)
    ts4 = create_trip_series(start_time=fixed_date)

    # Create sample time series data
    timestamps1 = pd.date_range(start='2023-01-01', periods=5, freq='D')
    timestamps2 = pd.date_range(start='2023-01-01', periods=5, freq='D')
    variables = ['var1', 'var2']
    data1 = np.random.rand(5, 2)
    data2 = np.random.rand(5, 2)

    metadata= TimeSeriesMetadata(1)
    # Create TimeSeries instances
    ts1 = TimeSeries(tsid='ts1', timestamps=timestamps1, variables=variables, data=data1,metadata=metadata)
    ts2 = TimeSeries(tsid='ts2', timestamps=timestamps2, variables=variables, data=data2,metadata=metadata)
    ts3.display_time_series()
    ts4.display_time_series()
    # Compute similarity measures
    print("Euclidean Distance:", ts3.euclidean_distance(ts4))
    print("Correlation Coefficient:", ts4.correlation_coefficient(ts3))
    print("Cosine Similarity:", ts3.cosine_similarity(ts4))
    print("DTW Distance:", ts4.dynamic_time_warping(ts3,'trip_count'))
    print('DTW MUltivariate', ts1.dtw_independent_multivariate(ts1))

