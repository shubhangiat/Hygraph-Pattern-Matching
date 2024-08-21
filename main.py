from datetime import datetime
import time
from watchdog.observers import Observer
from hygraph import HyGraph, Edge, PGNode
from fileProcessing import NodeFileHandler, EdgeFileHandler, HyGraphFileLoader


def connection_count_aggregate_function(graph, element_type, oid, attribute, date):
    if element_type == 'Edge':
        return len(list(graph.graph.neighbors(oid)))
    return 0


def count_edges_in_subgraph(subgraph, date):
    return sum(1 for _, _, edge_data in subgraph.view.edges(data=True) if edge_data['data'].start_time <= date)


if __name__ == "__main__":
    # Initialize the file loader with the directories and files
    loader = HyGraphFileLoader(
        nodes_folder='inputFiles/nodes',
        edges_folder='inputFiles/edges',
        subgraph_file='inputFiles/subgraphs.csv'
    )

    # Define a condition to filter edges
    def node_filter(node):
        print('node filter', node.properties.get('age', 0), node.label == 'Person' and int(node.properties.get('age', 0)) > 14 )
        return node.label == 'Person' and int(node.properties.get('age', 0)) > 14

    def edge_filter(edge):
        return edge.label == 'knows'
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
            'freq': 'W',  # Weekly frequency
            'direction': 'both'  # Direction for edges
        }
    }

    # Add the queries to the HyGraph instance
    loader.hygraph.add_query(query_node)
    # Start observing files for changes
    loader.start_file_observer()
    # Load all files and create the HyGraph instance
    loader.load_files()

    # Main event loop to handle updates
    try:
        while True:
            # Check if there have been updates to the graph
            if loader.hygraph.updated:
                loader.hygraph.display()
                loader.hygraph.set_updated(False)  # Reset the update flag after processing
            time.sleep(1)  # Sleep to reduce CPU usage
    except KeyboardInterrupt:
        loader.stop_observer()
