from datetime import datetime
import time
from watchdog.observers import Observer

from HyGraphFileLoaderBatch import HyGraphBatchProcessor
from hygraph import HyGraph, Edge, PGNode
from fileProcessing import NodeFileHandler, EdgeFileHandler, HyGraphFileLoader
import os

base_dir = os.path.dirname(os.path.abspath(__file__))


def connection_count_aggregate_function(graph, element_type, oid, attribute, date):
    if element_type == 'node':
        print("number of connections ",len(list(graph.graph.neighbors(oid))) )
        return len(list(graph.graph.neighbors(oid)))
    return 0


def count_edges_in_subgraph(subgraph, date):
    return sum(1 for _, _, edge_data in subgraph.view.edges(data=True) if edge_data['data'].start_time <= date)


if __name__ == "__main__":
    nodes_folder = os.path.join(base_dir, 'inputFiles', 'nodes')
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
        print("node filter here ok : ",int(node_obj.properties.get('age', 0)) > 14)
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

