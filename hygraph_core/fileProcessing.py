import os
import hashlib
from watchdog.events import FileSystemEventHandler
import pandas as pd
from watchdog.observers import Observer

from hygraph import HyGraph, Edge, PGNode, Subgraph, TimeSeries
from constraints import is_valid_membership

def file_hash(filepath):
    with open(filepath, "rb") as f:
        file_hash = hashlib.md5()
        chunk = f.read(4096)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(4096)
    return file_hash.hexdigest()

def get_label_from_file(file_path):
    # Extracts the label from the file name (without extension)
    return os.path.splitext(os.path.basename(file_path))[0]

class HyGraphFileLoader:
    def __init__(self, nodes_folder, edges_folder, subgraph_file ):
        self.hygraph = HyGraph()
        self.nodes_folder = nodes_folder
        self.edges_folder = edges_folder
        self.subgraph_file = subgraph_file
        self.observer = Observer()
        self.loading_complete = False  # Flag to track loading status

    def load_files(self):
        # Load nodes, edges, and subgraphs
        self.load_nodes()
        self.load_edges()
        self.process_subgraphs(self.subgraph_file)
        self.loading_complete = True  # Set flag to True after loading all files

    def load_nodes(self):
        node_file_paths = [os.path.join(self.nodes_folder, file) for file in os.listdir(self.nodes_folder) if file.endswith('.csv')]
        for node_file_path in node_file_paths:
            handler = NodeFileHandler(self.hygraph, node_file_path)
            handler.process_file()


    def load_edges(self):
        edge_file_paths = [os.path.join(self.edges_folder, file) for file in os.listdir(self.edges_folder) if file.endswith('.csv')]
        for edge_file_path in edge_file_paths:
            handler = EdgeFileHandler(self.hygraph, edge_file_path)
            handler.process_file()
    def is_loading_complete(self):
        return self.loading_complete

    def process_subgraphs(self, file_path):
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            subgraph_id, label, start_time, end_time = row['subgraph_id'], row['label'], row['start_time'], row['end_time']
            node_file_ids = row['node_file_ids'].split(',')
            edge_file_ids = row['edge_file_ids'].split(',')

            # Map external IDs (from file) to internal UUIDs
            node_internal_ids = [node.oid for node_id in node_file_ids
                                 for node in self.hygraph.graph.nodes(data=True)
                                 if node[1]['data'].node_id == node_id]

            edge_internal_ids = [edge.oid for edge_id in edge_file_ids
                                 for edge in self.hygraph.graph.edges(data=True)
                                 if edge[2]['data'].edge_id == edge_id]

            def filter_func(element):
                if isinstance(element, PGNode):
                    return element.oid in node_internal_ids
                elif isinstance(element, Edge):
                    return element.oid in edge_internal_ids
                return False

            subgraph = Subgraph(subgraph_id=subgraph_id, label=label, start_time=start_time, end_time=end_time, filter_func=filter_func)
            self.hygraph.add_subgraph(subgraph)

            # Update membership for nodes
            for node_id in node_internal_ids:
                node = self.hygraph.get_element('node', node_id)
                if is_valid_membership(node, start_time, end_time):
                    self.update_membership(node, subgraph_id, start_time, end_time)
                else:
                    print(f"Node {node_id} cannot be part of subgraph {subgraph_id} due to timeline mismatch.")

            # Update membership for edges
            for edge_id in edge_internal_ids:
                edge = self.hygraph.get_element('edge', edge_id)
                if is_valid_membership(edge, start_time, end_time):
                    self.update_membership(edge, subgraph_id, start_time, end_time)
                else:
                    print(f"Edge {edge_id} cannot be part of subgraph {subgraph_id} due to timeline mismatch.")

    def update_membership(self, element, subgraph_id, start_time, end_time):
        tsid = element.membership.get(subgraph_id) or self.hygraph.id_generator.generate_timeseries_id()

        if tsid not in self.hygraph.time_series:
            membership_series = TimeSeries(tsid, [start_time], ['membership'], [[subgraph_id]])
            self.hygraph.time_series[tsid] = membership_series
        else:
            membership_series = self.hygraph.time_series[tsid]
            membership_series.append_data(start_time, subgraph_id)

        if end_time:
            membership_series.append_data(end_time, None)

        element.membership[subgraph_id] = tsid
        print(f"Added subgraph {subgraph_id} to {type(element).__name__.lower()} {element.oid}'s membership from {start_time} to {end_time}.")

    def start_file_observer(self):
        # Iterate over each file in nodes and edges folders
        for node_file in os.listdir(self.nodes_folder):
            node_file_path = os.path.join(self.nodes_folder, node_file)
            if os.path.isfile(node_file_path):  # Check if it's a file, not a directory
                node_event_handler = NodeFileHandler(self.hygraph, node_file_path)
                self.observer.schedule(node_event_handler, path=node_file_path, recursive=False)

        for edge_file in os.listdir(self.edges_folder):
            edge_file_path = os.path.join(self.edges_folder, edge_file)
            if os.path.isfile(edge_file_path):  # Check if it's a file, not a directory
                edge_event_handler = EdgeFileHandler(self.hygraph, edge_file_path)
                self.observer.schedule(edge_event_handler, path=edge_file_path, recursive=False)

        self.observer.start()

    def stop_observer(self):
        self.observer.stop()
        self.observer.join()



class NodeFileHandler(FileSystemEventHandler):
    def __init__(self, hygraph, file_path):
        self.hygraph = hygraph
        self.file_path = file_path
        self.label = get_label_from_file(file_path)
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
        existing_nodes = {data['data'].node_id: data['data'] for _, data in self.hygraph.graph.nodes(data=True)}
        for _, row in df.iterrows():
            external_id, start_time, end_time = str(row['id']), row['start_time'], row.get('end_time')
            if external_id not in existing_nodes:
                new_node = PGNode(oid=self.hygraph.id_generator.generate_node_id(), label=self.label, start_time=row['start_time'], end_time=row.get('end_time'), node_id=external_id)
                new_node.properties.update(row.drop(['id', 'start_time', 'end_time']).to_dict())
                self.hygraph.add_node(new_node)
            else:
                node = existing_nodes[external_id]
                updated_properties = row.drop(['id', 'start_time', 'end_time']).to_dict()
                if node.properties != updated_properties:
                    node.properties.update(updated_properties)
                    print(f"Updated node {external_id} with new properties: {updated_properties}")


    def on_modified(self, event):
        print(f"Detected modification in {event.src_path}")
        # Ensure that only modifications to the actual data file are processed.
        if event.src_path.startswith("nodes/") and event.src_path.endswith(".csv"):
            print("Modification detected in a node file.")
            self.process_file()
        else:
            print(f"Ignored modification in {event.src_path}")


class EdgeFileHandler(FileSystemEventHandler):
    def __init__(self, hygraph, file_path):
        self.hygraph = hygraph
        self.file_path = file_path
        self.last_hash = None
        self.label = get_label_from_file(file_path)
        self.process_file()

    def process_file(self):
        current_hash = file_hash(self.file_path)
        if current_hash != self.last_hash:
            self.last_hash = current_hash
            try:
                df = pd.read_csv(self.file_path)
                self.update_graph(df)
            except Exception as e:
                print(f"Failed to process edge file: {e}")

    def update_graph(self, df):
        for _, row in df.iterrows():
            external_edge_id = str(row['id'])  # External edge ID from the file
            source_id, target_id, start_time, end_time = str(row['source_id']), str(row['target_id']), row['start_time'],  row.get('end_time')
            # Find the internal nodes by their external IDs
            source_node = next((node[1]['data'] for node in self.hygraph.graph.nodes(data=True) if node[1]['data'].node_id == source_id), None)
            target_node = next((node[1]['data'] for node in self.hygraph.graph.nodes(data=True) if node[1]['data'].node_id == target_id), None)

            if source_node and target_node:
                # Initialize edge_properties as an empty dictionary
                edge_properties = {}
                columns_to_drop = ['source_id', 'target_id', 'start_time', 'end_time']
                if len(row.drop(columns_to_drop)) > 0:
                    print(f"Processing edge from {source_id} to {target_id} with properties: ")
                    edge_properties = row.drop(columns_to_drop).to_dict()
                # Check if the edge already exists
                # Check if the edge already exists using internal IDs
                edge_exists = False
                for u, v, key, data in self.hygraph.graph.edges(data=True, keys=True):
                    existing_edge = data['data']
                    if external_edge_id == existing_edge.edge_id:
                        edge_exists = True
                        print('add property, element ', existing_edge.properties)
                        # Update the existing edge properties.py and external ID if they differ
                        if existing_edge.properties != edge_properties or existing_edge.edge_id != external_edge_id:

                            existing_edge.properties.update(edge_properties)
                            existing_edge.edge_id = external_edge_id  # Update the external edge ID
                            print(f"Updated edge from {source_id} to {target_id} with new properties: {edge_properties} and external ID: {external_edge_id}")
                        break  # No need to continue checking after the edge is found


                if not edge_exists:

                    edge = Edge(oid=self.hygraph.id_generator.generate_edge_id(),
                                source=source_node.oid,
                                target=target_node.oid,
                                label=self.label,
                                start_time=start_time,
                                end_time=end_time, edge_id=external_edge_id)

                    if edge_properties:
                        edge.properties.update(edge_properties)
                    self.hygraph.add_edge(edge)

    def on_modified(self, event):
        print(f"Detected modification in {event.src_path}")
        # Ensure that only modifications to the actual data file are processed.
        if event.src_path.startswith("edges/") and event.src_path.endswith(".csv"):
            print("Modification detected in an edge file.")
            self.process_file()
        else:
            print(f"Ignored modification in {event.src_path}")
