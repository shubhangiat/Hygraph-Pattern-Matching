from datetime import datetime

from hygraph import HyGraph, Edge
from generator import GraphDataGenerator

if __name__ == "__main__":
    hg = HyGraph()
    generator = GraphDataGenerator(hg)
    generator.generate_data()

    # Display the graph
    hg.display()


    # Attempt to add a property to a non-existent node
    try:
        hg.add_property(element_type='node', oid=100, property_key='humidity', tsid=2)
    except ValueError as e:
        print(e)

    # Attempt to add a property to a non-existent edge
    try:
        hg.add_property(element_type='edge', oid=100, property_key='weight', tsid=2)
    except ValueError as e:
        print(e)

    # Attempt to add a property to a non-existent subgraph
    try:
        hg.add_property(element_type='subgraph', oid='sg100', property_key='activity', tsid=1)
    except ValueError as e:
        print(e)
    # Add a new edge to test the update mechanism
    new_edge_id = generator.hygraph.id_generator.generate_edge_id()
    new_edge = Edge(oid=new_edge_id, source=generator.person_nodes[0], target=generator.person_nodes[1], label='connected', start_time=datetime(2020, 1, 1))
    hg.add_edge(new_edge)

    # Call the graph metrics evolution function to update memberships after adding the new edge
    hg.graph_metrics_evolution()