from datetime import datetime
import time
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from hygraph import HyGraph, Edge, PGNode

class NodeFileHandler(FileSystemEventHandler):
    def __init__(self, hygraph, file_path):
        self.hygraph = hygraph
        self.file_path = file_path
        self.process_file()

    def process_file(self):
        try:
            df = pd.read_csv(self.file_path)
            self.update_graph(df)
        except Exception as e:
            print(f"Failed to process node file: {e}")

    def update_graph(self, df):
        for _, row in df.iterrows():
            node_id, label, name = row['id'], row['label'], row['name']
            if not any(node.oid == str(node_id) for _, data in self.hygraph.graph.nodes(data=True) for node in [data['data']]):
                new_node = PGNode(oid=str(node_id), label=label, start_time=datetime.now())
                new_node.properties['name'] = name
                self.hygraph.add_node(new_node)

    def on_modified(self, event):
        if event.src_path == self.file_path:
            print(f"Node file {self.file_path} has been modified")
            self.process_file()

class EdgeFileHandler(FileSystemEventHandler):
    def __init__(self, hygraph, file_path):
        self.hygraph = hygraph
        self.file_path = file_path
        self.process_file()

    def process_file(self):
        try:
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
                edge = Edge(
                    oid=self.hygraph.id_generator.generate_edge_id(),
                    source=source_node.oid,
                    target=target_node.oid,
                    label=row.get('label', 'knows'),
                    start_time=datetime.now()
                )
                self.hygraph.add_edge(edge)

    def on_modified(self, event):
        if event.src_path == self.file_path:
            print(f"Edge file {self.file_path} has been modified")
            self.process_file()

def connection_count_aggregate_function(graph, element_type, oid, attribute, date):
    if element_type == 'node':
        return len(list(graph.graph.neighbors(oid)))
    return 0

if __name__ == "__main__":
    hygraph = HyGraph()
    hygraph.register_custom_metric('node', 'number_connections', connection_count_aggregate_function)

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
            time.sleep(10)
            hygraph.display()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
