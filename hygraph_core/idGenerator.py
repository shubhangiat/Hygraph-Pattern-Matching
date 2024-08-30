import uuid

class IDGenerator:
    def generate_node_id(self):
        return str(uuid.uuid4())

    def generate_edge_id(self):
        return str(uuid.uuid4())

    def generate_subgraph_id(self):
        return str(uuid.uuid4())

    def generate_timeseries_id(self):
        return str(uuid.uuid4())
