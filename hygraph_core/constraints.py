# Define all needed constraints
from datetime import datetime
# Placeholder for a date far in the future
FAR_FUTURE_DATE = datetime(2100, 12, 31, 23, 59, 59)
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
    #print(element.start_time,subgraph_start_time, parse_datetime(element.start_time), parse_datetime(subgraph_start_time))
    if parse_datetime(element.start_time) >  parse_datetime(subgraph_start_time):
        print(f"Element {element.oid} starts after the subgraph {subgraph_start_time}.")
        return False
    if (element.end_time and element.end_time!=FAR_FUTURE_DATE) and (subgraph_end_time and subgraph_end_time!=FAR_FUTURE_DATE) :
        if element.end_time < subgraph_end_time:
            print(f"Element {element.oid} ends before the subgraph {subgraph_end_time}.")
            return False
    print("marchan")
    return True

    # Convert string dates to datetime objects if necessary
def parse_datetime(datetime_str):
    if isinstance(datetime_str, str):
        print("value str : ", datetime_str)
        try:
            print(datetime_str)
            return datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        except (TypeError, ValueError):
            return None


