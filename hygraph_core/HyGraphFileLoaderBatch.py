import os
import pandas as pd
from datetime import datetime
from hygraph_core.hygraph import HyGraph


# Placeholder for a date far in the future
FAR_FUTURE_DATE = datetime(2100, 12, 31, 23, 59, 59)
class HyGraphBatchProcessor:
    def __init__(self, nodes_folder, edges_folder, subgraphs_folder,edges_membership, nodes_membership):
        self.hygraph = HyGraph()
        self.nodes_folder = nodes_folder
        self.edges_folder = edges_folder
        self.subgraphs_folder = subgraphs_folder
        self.edges_membership=edges_membership
        self.nodes_membership=nodes_membership
        self.node_batches = []  # List to accumulate node changes
        self.edge_batches = []  # List to accumulate edge changes
        self.subgraph_batches = []  # List to accumulate subgraph changes



    def load_files(self):
        # Load nodes, edges, and subgraphs
        self.load_nodes()
        self.load_edges()
        self.load_subgraphs()

    def load_nodes(self):
        node_file_paths = [os.path.join(self.nodes_folder, file) for file in os.listdir(self.nodes_folder) if
                           file.endswith('.csv')]
        for node_file_path in node_file_paths:
            label = os.path.basename(node_file_path).replace('.csv', '')  # Extract label from filename
            df = pd.read_csv(node_file_path)
            df['label'] = label  # Assign label to a new column in the DataFrame
            self.node_batches.append(df)

    def load_edges(self):
        edge_file_paths = [os.path.join(self.edges_folder, file) for file in os.listdir(self.edges_folder) if
                           file.endswith('.csv')]
        for edge_file_path in edge_file_paths:
            label = os.path.basename(edge_file_path).replace('.csv', '')  # Adjust if edge labels come from filenames
            df = pd.read_csv(edge_file_path)
            df['label'] = label  # Assign label to a new column in the DataFrame
            self.edge_batches.append(df)

    def load_subgraphs(self):
        subgraph_file_paths = [os.path.join(self.subgraphs_folder, file) for file in os.listdir(self.subgraphs_folder)
                               if file.endswith('.csv')]
        for subgraph_file_path in subgraph_file_paths:
            label = os.path.basename(subgraph_file_path).replace('.csv', '')  # Adjust if edge labels come from filenames
            df = pd.read_csv(subgraph_file_path)
            df['label'] = label  # Assign label to a new column in the DataFrame
            self.subgraph_batches.append(df)



    def process_batches(self):
        self.load_files()
        # Process all nodes first
        self.process_node_batches()
        # Then process all edges
        self.process_edge_batches()
        # Process all subgraphs
        self.process_subgraph_batches()
        # Process nodes and edges membership
        self.process_membership_data(self.edges_membership,"edge")
        self.process_membership_data(self.nodes_membership,"node")
        # After all nodes and edges are processed, trigger any necessary updates
        self.finalize_processing()


    def process_node_batches(self):
        for df in self.node_batches:
            existing_nodes = {data['data'].node_id: data['data'] for _, data in self.hygraph.graph.nodes(data=True)}
            for _, row in df.iterrows():
                external_id, start_time, end_time = str(row['id']), row['start_time'], row.get('end_time',FAR_FUTURE_DATE)
                end_time = self.end_time_config(end_time)
                if external_id not in existing_nodes:
                    self.hygraph.add_pgnode(oid=external_id, label=row['label'], start_time=pd.to_datetime(start_time), end_time=pd.to_datetime(end_time),properties=row.drop(['id', 'start_time', 'end_time']).to_dict())
                else:
                    node = existing_nodes[external_id]
                    updated_properties = row.drop(['id', 'start_time', 'end_time']).to_dict()
                    if node.properties != updated_properties:
                        node.properties.update(updated_properties)
                        print(f"Updated node {external_id} with new properties: {updated_properties}")

    def process_edge_batches(self):
        for df in self.edge_batches:
            for _, row in df.iterrows():
                external_edge_id = str(row['id'])
                source_id, target_id, start_time, end_time = str(row['source_id']), str(row['target_id']), row['start_time'], row.get('end_time',FAR_FUTURE_DATE)
                source_node = next((node[1]['data'] for node in self.hygraph.graph.nodes(data=True) if node[1]['data'].node_id == source_id), None)
                target_node = next((node[1]['data'] for node in self.hygraph.graph.nodes(data=True) if node[1]['data'].node_id == target_id), None)
                end_time=self.end_time_config(end_time)
                if source_node and target_node:
                    edge_exists = False
                    for u, v, key, data in self.hygraph.graph.edges(data=True, keys=True):
                        existing_edge = data['data']
                        if existing_edge.source == source_node.oid and existing_edge.target == target_node.oid:
                            edge_exists = True
                            if existing_edge.properties != row.drop(['source_id', 'target_id', 'start_time', 'end_time']).to_dict():
                                existing_edge.properties.update(row.drop(['source_id', 'target_id', 'start_time', 'end_time']).to_dict())
                                existing_edge.edge_id = external_edge_id
                            break

                    if not edge_exists:

                        self.hygraph.add_pgedge(oid=external_edge_id, source=source_node.oid, target=target_node.oid, label=row['label'], start_time=pd.to_datetime(start_time), end_time=pd.to_datetime(end_time),properties=row.drop(['source_id', 'target_id', 'start_time', 'end_time']).to_dict())
    def finalize_processing(self):
        print("Finalizing batch processing...")
        #self.hygraph.display()

    def process_subgraph_batches(self):
        for df in self.subgraph_batches:
            for _, row in df.iterrows():
                subgraph_id = row['id']
                label = row['label']
                start_time = pd.to_datetime(row['start_time'])
                end_time = pd.to_datetime(row['end_time'])
                properties = row.drop(['id', 'label', 'start_time', 'end_time']).to_dict()

                # Add the subgraph to the HyGraph
                self.hygraph.add_subgraph(subgraph_id=subgraph_id, label=label, start_time=pd.to_datetime(start_time), end_time=pd.to_datetime(end_time),properties=properties)
    """def process_subgraph_batches(self):
        # Create a dictionary to store subgraph start times
        subgraph_start_times = {}
        for df in self.subgraph_batches:
            for _, row in df.iterrows():
                subgraph_id = row['id']
                start_time = parse_datetime(row['start_time'])
                subgraph_start_times[subgraph_id] = start_time
        for df in self.subgraph_batches:
            for _, row in df.iterrows():
                label = row['label']
                subgraph_id, start_time, end_time = str(row['id']), row['start_time'], row.get('end_time',FAR_FUTURE_DATE)
                end_time = self.end_time_config(end_time)
                node_file_ids = row['node_file_ids'].split(' ')
                edge_file_ids = row['edge_file_ids'].split(' ')

                node_internal_ids = [id for node_id in node_file_ids
                                     for id, node_data in self.hygraph.graph.nodes(data=True)
                                     if str(node_data['data'].node_id) == str(node_id)]

                edge_internal_ids = [
                    key for edge_id in edge_file_ids
                    for _, _, key, edge_data in self.hygraph.graph.edges(keys=True, data=True)
                    if str(edge_data['data'].edge_id) == str(edge_id)  # Ensure IDs are compared as strings
                ]
                def filter_func(element):
                    if isinstance(element, PGNode):
                        return element.oid in node_internal_ids
                    elif isinstance(element, Edge):
                        return element.oid in edge_internal_ids
                    return False

                subgraph = Subgraph(subgraph_id=subgraph_id, label=label, start_time=start_time, end_time=end_time,
                                    filter_func=filter_func)
                self.hygraph.add_subgraph(subgraph)
                # Collect timestamps and subgraph IDs
                timestamps = []
                subgraph_ids = []
                # Update membership for nodes
                for node_id in node_internal_ids:
                    node = self.hygraph.get_element('node', node_id)
                    memberships = [subgraph_id for subgraph_id in subgraph_start_times if
                                   is_valid_membership(node, subgraph_start_times[subgraph_id], end_time)]
                    for ts in memberships:
                        timestamps.append(subgraph_start_times[ts])
                        subgraph_ids.append(ts)

                    self.hygraph.update_membership(node, timestamps, subgraph_ids)
                # Update membership for edges
                for edge_id in edge_internal_ids:
                    edge = self.hygraph.get_element('edge', edge_id)
                    memberships = [subgraph_id for subgraph_id in subgraph_start_times if
                                   is_valid_membership(edge, subgraph_start_times[subgraph_id], end_time)]
                    for ts in memberships:
                        timestamps.append(subgraph_start_times[ts])
                        subgraph_ids.append(ts)

                    self.hygraph.update_membership(edge, timestamps, subgraph_ids)"""

    '''def process_membership_data(self,file_path, element_type):
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df.sort_values(by='timestamp', inplace=True)  # Ensure entries are sorted by timestamp

        external_to_internal = self.from_external_to_internal_id(file_path, "id", element_type)
        print("all external ids", external_to_internal)
        for _, row in df.iterrows():
            external_ids = row['id'].split(' ')
            subgraph_ids = row['subgraph_id'].split(' ')
            timestamp = row['timestamp']
            action = row['action']
            print("external_ids",external_ids)
            for external_id in external_ids:
                external_id = external_id.strip()  # Clean up any extra whitespace
                # Ensure the external_id is mapped to an internal ID
                if external_id not in external_to_internal:
                    print('external id not in external_id',external_id)
                    internal_id = self.hygraph.id_generator.generate_node_id() if element_type == 'node'\
                        else self.hygraph.id_generator.generate_edge_id()
                else:internal_id = external_to_internal[external_id]
                if internal_id:
                    # Update the membership based on action
                    if action == 'add':
                        self.hygraph.add_membership(internal_id, timestamp, subgraph_ids,element_type)
                    elif action == 'remove':
                        self.hygraph.remove_membership(internal_id, timestamp, subgraph_ids, element_type)'''

    def process_membership_data(self, file_path, element_type):
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df.sort_values(by='timestamp', inplace=True)  # Ensure entries are sorted by timestamp

        print("Processing membership data from:", file_path)
        for _, row in df.iterrows():
            external_ids = row['id'].split(' ')
            subgraph_ids = row['subgraph_id'].split(' ')
            timestamp = row['timestamp']
            action = row['action']
            print("External IDs:", external_ids)
            for external_id in external_ids:
                external_id = external_id.strip()  # Clean up any extra whitespace
                # Since we're using external IDs directly, ensure the element exists
                if element_type == 'node':
                    if external_id not in self.hygraph.graph.nodes:
                        print(f"Node with ID '{external_id}' does not exist. Skipping.")
                        continue
                elif element_type == 'edge':
                    # For edges, we need to find the edge based on the external ID
                    edge_found = False
                    for u, v, k, data in self.hygraph.graph.edges(keys=True, data=True):
                        edge_data = data.get('data')
                        if edge_data and edge_data.oid == external_id:
                            edge_found = True
                            break
                    if not edge_found:
                        print(f"Edge with ID '{external_id}' does not exist. Skipping.")
                        continue
                else:
                    print(f"Invalid element type '{element_type}'. Skipping.")
                    continue

                # Update the membership based on action
                if action == 'add':
                    self.hygraph.add_membership(external_id, timestamp, subgraph_ids, element_type)
                elif action == 'remove':
                    self.hygraph.remove_membership(external_id, timestamp, subgraph_ids, element_type)
                else:
                    print(f"Unknown action '{action}' in row. Skipping.")

    # Utility Functions
    def end_time_config(self,end_time):
        if pd.isna(end_time):
            end_time = FAR_FUTURE_DATE
        return end_time

    def from_external_to_internal_id(self,file_path, id_column, element_type='node'):
        """ Load membership data from CSV and map external IDs to internal IDs. """
        membership_df = pd.read_csv(file_path)
        external_to_internal_ids = {}

        for _, row in membership_df.iterrows():
            external_ids = str(row[id_column]).split()  # Split IDs assuming they are separated by spaces
            for external_id in external_ids:
                if external_id not in external_to_internal_ids:
                    if element_type == 'node':
                        print('node here')
                        element = next((data['data'] for _, data in self.hygraph.graph.nodes(data=True) if data['data'].node_id == external_id),
                                       None)
                    else:  # id_type == 'edge'
                        print('edge here')
                        element = next(
                            (data['data'] for _, _, data in self.hygraph.graph.edges(data=True) if data['data'].edge_id == external_id), None)

                    internal_id = element.oid if element else (
                        self.hygraph.graph.id_generator.generate_node_id() if element_type == 'node' else self.hygraph.id_generator.generate_edge_id())
                    external_to_internal_ids[external_id] = internal_id
        return external_to_internal_ids