from datetime import datetime, timedelta
import random

from hygraph import PGNode, Edge


class GraphDataGenerator:
    def __init__(self, hygraph):
        self.hygraph = hygraph
        self.person_nodes = []
        self.post_nodes = []
    def generate_unique_time_series(self, base_date, num_points, variables):
        timestamps = [base_date + timedelta(days=i) for i in range(num_points)]
        data = [[random.randint(0, 100) for _ in variables] for _ in timestamps]
        return timestamps, data


    def generate_data(self, num_persons=10, num_posts=20, num_subgraphs=5):
        # Generate persons as nodes
        for i in range(num_persons):
            node_id = self.hygraph.id_generator.generate_node_id()
            start_time = datetime(2020, 1, 1)
            end_time = start_time + timedelta(days=random.randint(1, 365)) if random.random() > 0.5 else None
            person_node = PGNode(oid=node_id, label=f'Person_{i}', start_time=start_time, end_time=end_time)
            self.hygraph.add_node(person_node)
            self.person_nodes.append(node_id)

            # Add static property
            self.hygraph.add_property('node', node_id, 'age', random.randint(18, 70))

            # Add time-series property for number of connections
            self.hygraph.create_time_series_from_graph(
                element_type='node',
                oid=node_id,
                attribute='num_connections',
                start_date=start_time,
                aggregate_function=self.count_connections
            )
            # Generate posts as nodes
        for i in range(num_posts):
            node_id = self.hygraph.id_generator.generate_node_id()
            post_node = PGNode(oid=node_id, label=f'Post_{i}', start_time=datetime(2020, 1, 1))
            self.hygraph.add_node(post_node)
            self.post_nodes.append(node_id)

            # Add time-series property for number of likes
            self.hygraph.create_time_series_from_graph(
                element_type='node',
                oid=node_id,
                attribute='num_likes',
                start_date=post_node.start_time,
                aggregate_function=self.count_likes
            )
        # Generate connections between persons
        for i in range(num_persons):
            for j in range(i + 1, num_persons):
                if random.random() > 0.5:
                    edge_id = self.hygraph.id_generator.generate_edge_id()
                    start_time = datetime(2020, 1, 1)
                    end_time = start_time + timedelta(days=random.randint(1, 365)) if random.random() > 0.5 else None
                    edge = Edge(oid=edge_id, source=self.person_nodes[i], target=self.person_nodes[j], label='connected', start_time=start_time, end_time=end_time)
                    self.hygraph.add_edge(edge)

        # Generate likes (edges from persons to posts)
        for i in range(num_persons):
            for j in range(num_posts):
                if random.random() > 0.5:
                    edge_id = self.hygraph.id_generator.generate_edge_id()
                    start_time = datetime(2020, 1, 1)
                    end_time = start_time + timedelta(days=random.randint(1, 365)) if random.random() > 0.5 else None
                    edge = Edge(oid=edge_id, source=self.person_nodes[i], target=self.post_nodes[j], label='likes', start_time=start_time, end_time=end_time)
                    self.hygraph.add_edge(edge)

            # Generate subgraphs
        for i in range(num_subgraphs):
            subgraph_id = self.hygraph.id_generator.generate_subgraph_id()
            start_time = datetime(2020, 1, 1)
            end_time = start_time + timedelta(days=random.randint(1, 365)) if random.random() > 0.5 else None
            subgraph = Subgraph(subgraph_id=subgraph_id, label=f'Subgraph_{i}', start_time=start_time, end_time=end_time, filter_func=lambda x: True)
            self.hygraph.add_subgraph(subgraph)

            # Add time-series property for total number of people
            self.hygraph.create_time_series_from_graph(
                element_type='subgraph',
                oid=subgraph_id,
                attribute='total_num_ppl',
                start_date=start_time,
                aggregate_function=self.count_total_num_ppl
            )

        # Initial call to update memberships
        self.hygraph.graph_metrics_evolution()

    def count_connections(self, hygraph, element_type, oid, attribute, date):
        return sum(1 for edge in hygraph.graph.edges(data=True) if edge[2]['data'].source == oid or edge[2]['data'].target == oid)

    def count_likes(self, hygraph, element_type, oid, attribute, date):
        return sum(1 for edge in hygraph.graph.edges(data=True) if edge[2]['data'].target == oid and edge[2]['data'].label == 'likes')

    def count_total_num_ppl(self, hygraph, element_type, oid, attribute, date):
        return sum(1 for node in hygraph.subgraphs[oid]['view'].nodes)

    def display_generated_data(self):
        self.hygraph.display()
    #   variables = ['var1', 'var2']
    #  num_points = 10
    # subgraph_ids = [f'sg{k}' for k in range(num_subgraphs)]

    '''   for i in range(num_nodes):
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
            self.hygraph.add_property(element_type='subgraph', oid=f'sg{sg_id}', property_key='totalNumPpl', tsid=tsid)'''
