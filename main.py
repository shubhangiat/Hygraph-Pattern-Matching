from datetime import datetime
import time
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from hygraph import HyGraph, Edge, PGNode
import hashlib


def file_hash(filepath):
    with open(filepath, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(4096):
            file_hash.update(chunk)
    return file_hash.hexdigest()


class NodeFileHandler(FileSystemEventHandler):
    def __init__(self, hygraph, file_path):
        self.hygraph = hygraph
        self.file_path = file_path
        self.last_hash = None
        self.process_file()

    def process_file(self):
        current_hash = file_hash(self.file_path)
        if current_hash != self.last_hash:
            self.last_hash = current_hash
            try:
                df = pd.read_csv(self.file_path)
                self.update_graph(df)
            except Exception as e:
                print(f"Failed to process node file: {e}")

    def update_graph(self, df):
        existing_nodes = {node_id: data['data'] for node_id, data in self.hygraph.graph.nodes(data=True)}
        for _, row in df.iterrows():
            node_id, label, name = str(row['id']), row['label'], row['name']
            if node_id not in existing_nodes:
                new_node = PGNode(oid=node_id, label=label, start_time=datetime.now())
                new_node.properties['name'] = name
                self.hygraph.add_node(new_node)
            elif existing_nodes[node_id].properties.get('name') != name:
                existing_nodes[node_id].properties['name'] = name
                print(f"Updated node {node_id} with new name: {name}")

    def on_modified(self, event):
        print(f"Detected modification in {event.src_path}")
        if event.src_path == self.file_path:
            print(f"Node file {self.file_path} has been modified")
            self.process_file()


class EdgeFileHandler(FileSystemEventHandler):
    def __init__(self, hygraph, file_path):
        self.hygraph = hygraph
        self.file_path = file_path
        self.last_hash = None
        self.process_file()

    def process_file(self):
        try:
            current_hash = file_hash(self.file_path)
            if current_hash != self.last_hash:
                self.last_hash = current_hash
                df = pd.read_csv(self.file_path)
                self.update_graph(df)
        except Exception as e:
            print(f"Failed to process edge file: {e}")

    def update_graph(self, df):
        for _, row in df.iterrows():
            source_id, target_id = str(row['source_id']), str(row['target_id'])
            source_node = self.hygraph.get_element('node', source_id)
            target_node = self.hygraph.get_element('node', target_id)

            if source_node and target_node:
                # Check if the edge already exists
                if not any(edge for edge in self.hygraph.graph.edges(data=True) if edge[2]['data'].source == source_id and edge[2]['data'].target == target_id):
                    edge = Edge(
                        oid=self.hygraph.id_generator.generate_edge_id(),
                        source=source_node.oid,
                        target=target_node.oid,
                        label=row.get('label', 'knows'),
                        start_time=datetime.now()
                    )
                    self.hygraph.add_edge(edge)
                else:
                    # Optional: Update existing edge if needed
                    pass

    def on_modified(self, event):
        print(f"Detected modification in {event.src_path}")
        # Ensure that only modifications to the actual data file are processed.
        if event.src_path.endswith("nodes.csv"):
            print("Modification detected in the actual nodes.csv file.")
            self.process_file()
        else:
            print(f"Ignored modification in {event.src_path}")


def connection_count_aggregate_function(graph, element_type, oid, attribute, date):
    if element_type == 'node':
        return len(list(graph.graph.neighbors(oid)))
    return 0


if __name__ == "__main__":
    hygraph = HyGraph()
    hygraph.register_custom_metric('node', 'number_connections', connection_count_aggregate_function, "knows")

    node_file_path = 'nodes.csv'
    edge_file_path = 'edges.csv'

    node_event_handler = NodeFileHandler(hygraph, node_file_path)
    edge_event_handler = EdgeFileHandler(hygraph, edge_file_path)

    observer = Observer()
    observer.schedule(node_event_handler, path='.', recursive=False)
    observer.schedule(edge_event_handler, path='.', recursive=False)
    observer.start()

    try:
        while True:
            if hygraph.updated:
                hygraph.display()
            time.sleep(1)  # Short sleep to reduce CPU usage
    except KeyboardInterrupt:
        observer.stop()
    observer.join()