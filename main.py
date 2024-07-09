from hygraph import HyGraph
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