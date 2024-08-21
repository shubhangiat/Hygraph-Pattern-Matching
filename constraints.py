# Define all needed constraints

def is_valid_membership(element, subgraph_start_time, subgraph_end_time):
    """
    Validate if an element (node or edge) can be part of a subgraph based on the timeline.

    Parameters:
    - element: The node or edge to validate.
    - subgraph_start_time: The start time of the subgraph.
    - subgraph_end_time: The end time of the subgraph.

    Returns:
    - bool: True if the element can be part of the subgraph, False otherwise.
    """
    # Check if the element's start_time and end_time are within the subgraph's timeline
    if element.start_time > subgraph_start_time:
        print(f"Element {element.oid} starts after the subgraph {subgraph_start_time}.")
        return False
    if element.end_time and subgraph_end_time and element.end_time < subgraph_end_time:
        print(f"Element {element.oid} ends before the subgraph {subgraph_end_time}.")
        return False
    return True
