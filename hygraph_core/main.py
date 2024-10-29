from datetime import datetime, timedelta
import time
from watchdog.observers import Observer

from HyGraphFileLoaderBatch import HyGraphBatchProcessor
from hygraph import HyGraph, Edge, PGNode, HyGraphQuery
from fileProcessing import NodeFileHandler, EdgeFileHandler, HyGraphFileLoader
import os

from hygraph_core.timeseries_operators import TimeSeries

base_dir = os.path.dirname(os.path.abspath(__file__))


def connection_count_aggregate_function(graph, element_type, oid, attribute, date):
    if element_type == 'node':
        print("number of connections ", len(list(graph.graph.neighbors(oid))))
        return len(list(graph.graph.neighbors(oid)))
    return 0


def count_edges_in_subgraph(subgraph, date):
    return sum(1 for _, _, edge_data in subgraph.view.edges(data=True) if edge_data['data'].start_time <= date)


if __name__ == "__main__":
    '''nodes_folder = os.path.join(base_dir, 'inputFiles', 'nodes')
    edges_folder = os.path.join(base_dir, 'inputFiles', 'edges')
    subgraph_folder = os.path.join(base_dir, 'inputFiles', 'subgraphs')
    edges_membership_path = base_dir+"/inputFiles/edge_membership.csv"
    nodes_membership_path = base_dir+"/inputFiles/node_membership.csv"

    # Initialize the file loader with the directories and files
    loader = HyGraphBatchProcessor(
        nodes_folder=nodes_folder,
        edges_folder=edges_folder,
        subgraphs_folder=subgraph_folder,
        edges_membership=edges_membership_path,
        nodes_membership=nodes_membership_path
    )

    # Define a condition to filter edges
    def node_filter(node):
        node_obj = node['data']
        print("node filter here ok : ", int(node_obj.properties.get('age', 0)) > 14)
        return node_obj.label == 'Person' and int(node_obj.properties.get('age', 0)) > 14

    def edge_filter(edge):
        edge_obj = edge['data']
        return edge_obj.label == 'Knows'
    # Define the first query for nodes
    query_node = {
        'element_type': 'node',
        'node_filter': node_filter,
        'edge_filter': edge_filter,
        'time_series_config': {
            'start_date': datetime(2023, 1, 1),
            'end_date': datetime(2023, 12, 31),
            'attribute': 'number_connections',
            'aggregate_function': connection_count_aggregate_function,
            'freq': 'M',  # Weekly frequency
            'direction': 'both'  # Direction for edges
        }
    }
    loader.process_batches()  # Process all batches
    # Add the queries to the HyGraph instance
    loader.hygraph.add_query(query_node)
    loader.hygraph.display()
    # Batch processing trigger based on user input
    while True:
        try:
            user_input = input("Type 'run' to start batch processing: ")
            if user_input == 'run':
                print("Starting batch processing...")
                loader.process_batches()  # Process all batches
                loader.hygraph.batch_process()
                loader.hygraph.display()
                print("Batch processing completed.")
        except KeyboardInterrupt:
            print("Batch processing terminated.")
            break


#out of files :
    # Sample data for time series
    timestamps = ['2024-10-01', '2024-10-02', '2024-10-03']
    variables = ['temperature']
    data = [[25], [27], [28]]

    # Add a time series in the hygraph
    graph_element.add_temporal_property('temperature', timestamps, variables, data) '''

    hygraph = HyGraph()  # Initialize an empty HyGraph instance

    # Add mock PGNode stations with static properties including 'capacity'
    node1 = hygraph.add_pgnode(oid=1, label='Station', start_time=datetime.now()- timedelta(hours=7),
                               properties={'capacity': 100, 'name': 'Station A'})
    node2=hygraph.add_pgnode(oid=2, label='Station', start_time=datetime.now()- timedelta(hours=8),
                       properties={'capacity': 40, 'name': 'Station B'})
    node3=hygraph.add_pgnode(oid=3, label='Station', start_time=datetime.now()- timedelta(hours=7),
                       properties={'capacity': 60, 'name': 'Station C'})

    try:
        data = hygraph.graph.nodes[1]
        print("Data keys for node:", data.keys())
        print("Specific data content:", {k: data[k] for k in data.keys()})
    except Exception as e:
        print(f"Error accessing data for node {1}: {e}")



    edge1=hygraph.add_pgedge(oid=4,source=1,target=2,label='Trip', start_time=datetime.now() - timedelta(hours=4))
    edge2 = hygraph.add_pgedge(oid=5, source=2, target=3, label='Trip', start_time=datetime.now() - timedelta(hours=3))
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
    ts_in_degree = node_degree_history = hygraph.get_node_degree_over_time(node_id=1, degree_type='in',
                                                                           return_type='history')
    ts_out_degree = hygraph.get_node_degree_over_time(node_id=1, degree_type='out', return_type='history')
    ts_in_degree.aggregate_time_series_cumulative(ts_out_degree,'both_degree').display_time_series()

    subgraph_original=hygraph.add_subgraph('manhattan', label='Manhattan Subgraph',start_time=datetime(2023, 1, 1))
    subgraph_original['data'].add_static_property("text",230,hygraph)
    t1 = datetime.now() - timedelta(hours=3)
    manhattan_station_ids = [1,2,3]  # IDs of stations in Manhattan at time t1
    manhattan_edge_ids = [4,5]  # IDs of edges (trips) in Manhattan at time t1
    # Add memberships for nodes
    for node_id in manhattan_station_ids:
        hygraph.add_membership(node_id, t1, ['manhattan'], 'node')

    # Add memberships for edges
    for edge_id in manhattan_edge_ids:
        hygraph.add_membership(edge_id, t1, ['manhattan'], 'edge')
    t2 = datetime.now()
    nodes_to_remove = [1]  # IDs of nodes to remove
    edges_to_remove = [4]  # IDs of edges to remove
    for node_id in nodes_to_remove:
        hygraph.remove_membership(node_id, t2, ['manhattan'], 'node')

    for edge_id in edges_to_remove:
        hygraph.remove_membership(edge_id, t2, ['manhattan'], 'edge')
    # At time t1
    subgraph_t1 = hygraph.get_subgraph_at('manhattan', t1)
    # Display subgraph_t1...

    # At time t2
    subgraph_t2 = hygraph.get_subgraph_at('manhattan', t2)

    hygraph.display_subgraph('manhattan',t1)

    # Display subgraph_t2...
