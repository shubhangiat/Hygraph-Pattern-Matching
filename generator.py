from datetime import datetime, timedelta
import random

class GraphDataGenerator:
    def __init__(self, hygraph):
        self.hygraph = hygraph

    def generate_unique_time_series(self, base_date, num_points, variables):
        timestamps = [base_date + timedelta(days=i) for i in range(num_points)]
        data = [[random.randint(0, 100) for _ in variables] for _ in timestamps]
        return timestamps, data

    def generate_membership_time_series(self, base_date, num_points, subgraph_ids):
        timestamps = [base_date + timedelta(days=i) for i in range(num_points)]
        data = [[", ".join(random.sample(subgraph_ids, random.randint(1, len(subgraph_ids))))] for _ in timestamps]
        return timestamps, data

    def generate_data(self, num_nodes=10, num_edges=20, num_subgraphs=5):
        for i in range(num_nodes):
            start_time = datetime(2020, 1, 1)
            end_time = start_time + timedelta(days=random.randint(1, 365)) if random.random() > 0.5 else None
            self.hygraph.add_vertex(oid=i, label=f'Person{i}', start_time=start_time, end_time=end_time)

        edge_count = 0
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if edge_count < num_edges:
                    start_time = datetime(2020, 1, 2)
                    end_time = start_time + timedelta(days=random.randint(1, 365)) if random.random() > 0.5 else None
                    self.hygraph.add_edge(oid=edge_count, source=i, target=j, label='relatedTo', start_time=start_time, end_time=end_time)
                    edge_count += 1

        variables = ['var1', 'var2']
        num_points = 10
        subgraph_ids = [f'sg{k}' for k in range(num_subgraphs)]

        for i in range(num_nodes):
            base_date = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 365))
            timestamps, data = self.generate_unique_time_series(base_date, num_points, variables)
            self.hygraph.create_time_series(tsid=i, timestamps=timestamps, variables=variables, data=data)
            self.hygraph.add_property(element_type='node', oid=i, property_key='numberOfPosts', tsid=i)

            membership_timestamps, membership_data = self.generate_membership_time_series(base_date, num_points, subgraph_ids)
            self.hygraph.create_time_series(tsid=f"m{i}", timestamps=membership_timestamps, variables=['subgraph'], data=membership_data)
            self.hygraph.add_membership(element_type='node', oid=i, tsid=f"m{i}")

        for i in range(edge_count):
            tsid = i + num_nodes
            base_date = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 365))
            timestamps, data = self.generate_unique_time_series(base_date, num_points, variables)
            self.hygraph.create_time_series(tsid=tsid, timestamps=timestamps, variables=variables, data=data)
            self.hygraph.add_property(element_type='edge', oid=i, property_key='numberOfLikedPosts', tsid=tsid)

            membership_timestamps, membership_data = self.generate_membership_time_series(base_date, num_points, subgraph_ids)
            self.hygraph.create_time_series(tsid=f"me{i}", timestamps=membership_timestamps, variables=['subgraph'], data=membership_data)
            self.hygraph.add_membership(element_type='edge', oid=i, tsid=f"me{i}")

        for sg_id in range(num_subgraphs):
            start_time = datetime(2020, 1, 1)
            end_time = start_time + timedelta(days=random.randint(1, 365)) if random.random() > 0.5 else None
            self.hygraph.add_subgraph(subgraph_id=f'sg{sg_id}', label=f'Subgraph{sg_id}', start_time=start_time, end_time=end_time)

            tsid = sg_id + num_nodes + edge_count
            base_date = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 365))
            timestamps, data = self.generate_unique_time_series(base_date, num_points, variables)
            self.hygraph.create_time_series(tsid=tsid, timestamps=timestamps, variables=variables, data=data)
            self.hygraph.add_property(element_type='subgraph', oid=f'sg{sg_id}', property_key='totalNumPpl', tsid=tsid)
