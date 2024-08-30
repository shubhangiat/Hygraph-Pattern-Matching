# custom_aggregation_functions.py

# Example aggregate function: Counts the number of connections for a node.
def connection_count_aggregate_function(hygraph, element_type, oid, attribute, date):
    if element_type == 'node':
        node = hygraph.get_element('node', oid)
        edges = hygraph.graph.edges(node.oid, data=True)
        return sum(1 for _, _, edge_data in edges if edge_data['data'].start_time <= date)
    return 0

# Example aggregate function: Calculates the average strength of connections.
def average_strength_aggregate_function(hygraph, element_type, oid, attribute, date):
    if element_type == 'edge':
        edge = hygraph.get_element('edge', oid)
        return edge.properties.get('strength', 0)
    return 0

# Example aggregate function: Calculates the average age of nodes.
def average_age_aggregate_function(hygraph, element_type, oid, attribute, date):
    if element_type == 'node':
        node = hygraph.get_element('node', oid)
        return int(node.properties.get('age', 0))
    return 0

# Users can add their custom aggregation functions below.
# Example:
# def custom_aggregate_function(hygraph, element_type, oid, attribute, date):
#     # Custom logic here
#     return custom_value

