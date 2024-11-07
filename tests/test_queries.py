import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

from hygraph_core.graph_operators import PGEdge
from hygraph_core.hygraph import HyGraphQuery, HyGraph
from hygraph_core.timeseries_operators import TimeSeries, TimeSeriesMetadata


# Assuming HyGraphQuery and all necessary classes are already imported
# Mock data and HyGraph setup will be included

class TestHyGraphQuery(unittest.TestCase):

    def setUp(self):
        """
        Set up the mock HyGraph instance for testing.
        """
        self.hygraph = HyGraph()  # Initialize an empty HyGraph instance

        fixed_date = datetime(2023, 1, 1,8, 0, 0)
        self.start_time = fixed_date
        self.end_time = fixed_date + timedelta(minutes=20)
        self.start_time2=self.start_time+timedelta(days=10)

        # Add mock PGNode stations with static properties including 'capacity'
        node1=self.hygraph.add_pgnode(oid=1, label='Station', start_time=self.start_time,
                                properties={'capacity': 100, 'name': 'Station A'},)
        node2=self.hygraph.add_pgnode(oid=2, label='Station', start_time=self.start_time,
                                properties={'capacity': 40, 'name': 'Station B'})
        node3=self.hygraph.add_pgnode(oid=3, label='Station', start_time=self.start_time2,
                                properties={'capacity': 60, 'name': 'Station C'})


        def create_timeseries_nodes_decrement(start_time,length=13):
            timestamps = [start_time + timedelta(minutes=5 * i) for i in range(length)]
            # Set an initial high count and ensure a strictly decreasing sequence with unique values
            initial_count = np.random.randint(30, 40)  # Start with a high value
            bikeAvailable = []
            current_count = initial_count

            for _ in range(length):
                decrement = np.random.randint(1, 5)  # Ensure at least a decrement of 1
                current_count -= decrement
                if current_count <= 0:
                    current_count = 1  # Ensure no values go below 1 to maintain uniqueness and non-zero counts
                bikeAvailable.append(current_count)
            data = np.array(bikeAvailable).reshape((length, 1))
            metadata = TimeSeriesMetadata(owner_id=None)
            ts1 = self.hygraph.add_time_series(timestamps=timestamps, variables=['BikeAvailability'], data=data,
                                               metadata=metadata)
            return ts1

        def create_timeseries_nodes_increment(start_time, length=13):
            timestamps = [start_time + timedelta(minutes=5 * i) for i in range(length)]
            # Set an initial high count and ensure a strictly decreasing sequence with unique values

            # Set an initial low count and ensure a strictly increasing sequence with unique values
            initial_count = np.random.randint(1, 5)  # Start with a low value
            bikeAvailable = []
            current_count = initial_count

            for _ in range(length):
                increment = np.random.randint(1, 5)  # Ensure at least an increment of 1
                current_count += increment  # Add increment to make it an increasing series
                bikeAvailable.append(current_count)
            data = np.array(bikeAvailable).reshape((length, 1))
            metadata = TimeSeriesMetadata(owner_id=None)
            ts1 = self.hygraph.add_time_series(timestamps=timestamps, variables=['BikeAvailability'], data=data,
                                               metadata=metadata)
            return ts1
        ts_node1=create_timeseries_nodes_decrement(self.start_time)
        ts_node2=create_timeseries_nodes_decrement(self.start_time)
        ts_node3=create_timeseries_nodes_increment(self.start_time2)
        node1.add_temporal_property("bikeavailable", ts_node1, self.hygraph)
        node2.add_temporal_property("bikeavailable", ts_node2, self.hygraph)
        node3.add_temporal_property("bikeavailable", ts_node3, self.hygraph)

        start_time = datetime(2023, 1, 1, 8, 0, 0)  # Example start time
        end_time = datetime(2023, 1, 1, 10, 0, 0)  # Example end time

        # Function to create a time series for trip counts
        def create_trip_counter_series(start_time, length=10):
            timestamps = [start_time + timedelta(minutes=5 * i) for i in range(length)]
            trip_counts = np.random.randint(5, 20, size=length)  # Random counts between 5 and 20
            data = np.array(trip_counts).reshape((length, 1))
            metadata = TimeSeriesMetadata(owner_id=None)
            ts1=self.hygraph.add_time_series(timestamps=timestamps, variables=['trip_count'], data=data,metadata=metadata)
            return ts1
        def create_trip_multivariate_timeseries(start_time,id,i=1,j=1,length=10,):
            # Create multivariate timeseries
            timestamps = pd.date_range(start_time, periods=10, freq="5T")
            # Generate random data for each variable
            trip_counts = np.random.randint(5, 20, size=(length,))
            distances = np.random.randint(1, 10, size=(length,))
            data =np.stack([trip_counts*i,distances*j],axis=-1)

            metadata = TimeSeriesMetadata(owner_id=id)
            ts1=self.hygraph.add_time_series(timestamps=timestamps, variables=["electric_bike", "classic_bike"], data=data,metadata=metadata)
            print('inserted time series',ts1.display_time_series())
            return ts1

        trip_counter_series = create_trip_counter_series(start_time)
        trip_counter_series2=create_trip_counter_series(start_time)

        data_array_edge2 = create_trip_multivariate_timeseries(start_time,id=4)
        data_array_edge1=create_trip_multivariate_timeseries(start_time,5,2,2,)
        self.edge1=self.hygraph.add_pgedge(oid=4, source=1, target=2, label='Trip', start_time=datetime.now(), properties={'bike_type': 'electric', 'trip_counter': trip_counter_series,'distance':800,'multivariate':data_array_edge1})
        self.edge2=self.hygraph.add_pgedge(oid=5, source=2, target=3, label='Trip', start_time=datetime.now(), properties={'trip_counter': trip_counter_series2,'distance':1000,'multivariate':data_array_edge2})
        self.edge3=self.hygraph.add_pgedge(oid=6, source=1, target=2, label='Other', start_time=datetime.now(), properties={'user': 'member', })



    def test_station_capacity_above_50(self):
        query = HyGraphQuery(self.hygraph)
        results = (
            query
            .match_node(alias='station', label='Station', node_type='PGNode')
            .where(lambda node: node.get_static_property('capacity') > 50)
            .return_(
                station_id=lambda n: n['station'].getId(),
                station_name=lambda n: n['station'].get_static_property('name'),
                capacity=lambda n: n['station'].get_static_property('capacity')
            )
            .execute()
        )
        # Expected result (based on the setup)
        expected_results = [
            {'station_id': 1, 'station_name': 'Station A'},  # Capacity 100
            {'station_id': 3, 'station_name': 'Station C'}  # Capacity 60
        ]
        # Verify results
        self.assertEqual(len(results), 2)  # Only 'Station A' and 'Station C' should match
        self.assertEqual(len(results), len(expected_results))  # Check if the number of results matches
        for result, expected in zip(results, expected_results):
            self.assertEqual(result['station_id'], expected['station_id'])  # Check station_id
            self.assertEqual(result['station_name'], expected['station_name'])  # Check station_name

    def test_active_trips_between_two_stations(self):
        self.edge1.get_temporal_property('trip_counter')
        query = HyGraphQuery(self.hygraph)
        results = (
            query
            .match_node(alias='stationA', label='Station', node_type='PGNode')
            .where(lambda node: node.get_static_property('name') == 'Station A')
            .match_edge(alias='trip', label='Trip', edge_type='PGEdge')
            .match_node(alias='stationB', label='Station', node_type='PGNode')
            .where(lambda node: node.get_static_property('name') == 'Station B')
            .connect('stationA', 'trip', 'stationB')
            .return_(
                peak_trip_data=lambda n: {
                    'peak_trip_count': (
                        max_count := n['trip'].get_temporal_property('trip_counter', 0).apply_aggregation(
                            'max', start_time=self.start_time, end_time=self.end_time
                        )),
                    'peak_trip_timestamp': n['trip'].get_temporal_property('trip_counter', 0).get_timestamp_at_value(
                        max_count)
                },
                from_station=lambda n: n['stationA'].get_static_property('name'),
                to_station=lambda n: n['stationB'].get_static_property('name')
            )
            .execute()
        )

        for result in results:
            peak_trip_data = result['peak_trip_data']
            print(f"From Station: {result['from_station']}")
            print(f"To Station: {result['to_station']}")
            print(f"Peak Trip Count: {peak_trip_data['peak_trip_count']}")
            print(f"Peak Trip Timestamp: {peak_trip_data['peak_trip_timestamp']}")
        # Assertion: Only one result returned

    def test_aggregate_total_trips(self):
        # Create and execute the query
        query = HyGraphQuery(self.hygraph)
        results = (
            query
            .match_node(alias='station', label='Station', node_type='PGNode')
            .match_edge(alias='trip', label='Trip', edge_type='PGEdge')
            .connect('station', 'trip', 'station')
            .group_by('station')
            .aggregate(
                alias='trip',  # Specify the alias for which we want to aggregate
                property_name='trip_counter',  # The property to sum up
                method='sum'  # Aggregation method (sum in this case)
            )
            .return_(
                station_name=lambda n: n['station'].get_static_property('name'),
                total_trips=lambda n: n['trip_counter'].display_time_series())
            .execute()
        )

        # Process and print results
        for result in results:
            print(f"Station: {result['station_name']}")
            print(f"Total Trips: {result['total_trips']}")
    def test_aggregate_time_series_by_station(self):

        print('timeseries edge1',self.edge1.get_static_property('distance'))
        print('timeseries edge2', self.edge2.get_static_property('distance'))
        print('multivariate',self.edge2.get_temporal_property('multivariate'))
        query = HyGraphQuery(self.hygraph)
        results = (
            query
            .match_node(alias='station', label='Station', node_type='PGNode')
            .match_edge(alias='trip', label='Trip', edge_type='PGEdge')
            .match_node(alias='other_station', label='Station', node_type='PGNode')
            .connect('station', 'trip', 'other_station')
            .group_by('station')
            .aggregate('trip', 'trip_counter', method='sum', direction='both')  # Aggregate time series property
            .aggregate('trip', 'distance', method='mean', direction='both')  # Aggregate static property
            .aggregate('trip', 'multivariate', method='sum',
                       direction='both')  # Aggregate the multivariate time series # Aggregate static property
            .return_(
                station_name=lambda n: n['station'].get_static_property('name'),
                total_trips=lambda n: n['trip_counter'].display_time_series(),  # Aggregated TimeSeries result
                total_distance=lambda n: n['distance'],  # Aggregated static result
                aggregated_multivar_ts=lambda n: n['multivariate'].display_time_series()
                # Aggregated multivariate TimeSeries
            )
            .execute()
        )

        for result in results:
            print(f"Station Name: {result['station_name']}")
            print(f"Aggregated Time Series for Trips: {result['total_trips']}")
            print(f"Aggregated distances Trips: {result['total_distance']}")  # Should display TimeSeries output
            print(f"Multivariate: {result['aggregated_multivar_ts']}")
            '''print(f"Aggregated distances Trips: {result['total_distance']}")# Should display TimeSeries output'''

    def test_decreasing_bike_availability(self):
        # Define a function to identify nodes with decreasing bike availability
        def has_decreasing_availability(node):
            time_series = node.get_temporal_property('bikeavailable')
            data = time_series.data.to_pandas()
            # Print the data to confirm structure
            print('data structure: ', data)
            recent_data = data['BikeAvailability'].tail(5)  # Consider the last 5 data points
            return recent_data.is_monotonic_decreasing

        query = HyGraphQuery(self.hygraph)
        results = (
            query
            .match_node(alias='station', label='Station', node_type='PGNode')
            .where(has_decreasing_availability)
            .return_(
                station_name=lambda n: n['station'].get_static_property('name'),
                last_availability=lambda n: n['station'].get_temporal_property('bikeavailable').last_value()[1][0]
            )
            .execute()
        )
        # Test that results include only stations with decreasing availability
        for result in results:
            print(f"Station Name: {result['station_name']}")
            print(f"Decreasing bike availability: {result['last_availability']}")

    def test_decreasing_bike_availability_neighbors(self):
        # Assume `has_decreasing_availability` is a function that checks if a node's time series is decreasing

        def has_decreasing_availability(node):
            time_series = node.get_temporal_property('bikeavailable')
            data = time_series.data.to_pandas()
            # Print the data to confirm structure
            print('data structure: ', data)
            recent_data = data['BikeAvailability'].tail(5)  # Consider the last 5 data points
            return recent_data.is_monotonic_decreasing

        query = HyGraphQuery(self.hygraph)
        results = (
            query
            .match_node(alias='station', label='Station', node_type='PGNode')
            .where(has_decreasing_availability)  # Filter nodes by decreasing availability
            .return_(
                station_name=lambda n: n['station'].get_static_property('name'),
                neighbors=lambda n: [neighbor.get_id() for neighbor in n['station'].get_neighbors(self.hygraph)]  # Get neighbor IDs
            )
            .execute()
        )

        for result in results:
            print(f"Station: {result['station_name']}, Neighbors: {result['neighbors']}")

    from datetime import datetime

    def test_shortest_path_with_high_availability(self):
        fixed_date = datetime(2023, 1, 1, 8, 0, 0)

        station_A = self.hygraph.add_pgnode(
            oid=10, label='Station', start_time=fixed_date,
            properties={'capacity': 100, 'name': 'Station A'}
        )

        station_B = self.hygraph.add_pgnode(
            oid=12, label='Station', start_time=fixed_date,
            properties={'capacity': 80, 'name': 'Station B'}
        )

        station_C = self.hygraph.add_pgnode(
            oid=13, label='Station', start_time=fixed_date,
            properties={'capacity': 60, 'name': 'Station C'}
        )

        def create_multivariate_bike_availability(start_time, length=13):
            timestamps = [start_time + timedelta(minutes=5 * i) for i in range(length)]
            electric_bike_counts = np.random.randint(5, 30, size=length)
            classic_bike_counts = np.random.randint(10, 50, size=length)
            data = np.column_stack([electric_bike_counts, classic_bike_counts])
            metadata = TimeSeriesMetadata(owner_id=None)

            # Create multivariate time series for both electric and classic bikes
            return self.hygraph.add_time_series(
                timestamps=timestamps, variables=['electric_bike', 'classic_bike'], data=data, metadata=metadata
            )

        # Add time series data to each station node
        station_A.add_temporal_property("bike_availability", create_multivariate_bike_availability(fixed_date), self.hygraph)
        station_B.add_temporal_property("bike_availability", create_multivariate_bike_availability(fixed_date), self.hygraph)
        station_C.add_temporal_property("bike_availability", create_multivariate_bike_availability(fixed_date), self.hygraph)

        def create_multivariate_bike_availability(start_time, length=13):
            timestamps = [start_time + timedelta(minutes=5 * i) for i in range(length)]
            electric_bike_counts = np.random.randint(5, 30, size=length)
            classic_bike_counts = np.random.randint(10, 50, size=length)
            data = np.column_stack([electric_bike_counts, classic_bike_counts])
            metadata = TimeSeriesMetadata(owner_id=None)

            # Create multivariate time series for both electric and classic bikes
            return self.hygraph.add_time_series(
                timestamps=timestamps, variables=['electric_bike', 'classic_bike'], data=data, metadata=metadata
            )


        def create_trip_series(start_time, length=10):
            timestamps = [start_time + timedelta(minutes=5 * i) for i in range(length)]
            trip_counts = np.random.randint(5, 20, size=length)
            distances = np.random.randint(500, 1500, size=length)
            data = np.column_stack([trip_counts, distances])
            metadata = TimeSeriesMetadata(owner_id=None)

            return self.hygraph.add_time_series(
                timestamps=timestamps, variables=['trip_count', 'distance'], data=data, metadata=metadata
            )

        # Define trip edges between stations with multivariate time series data
        trip_A_B = self.hygraph.add_pgedge(
            oid=101, source=station_A.getId(), target=station_B.getId(), label='Trip',
            start_time=fixed_date, properties={'bike_type': 'electric', 'trip_data': create_trip_series(fixed_date)}
        )

        trip_B_C = self.hygraph.add_pgedge(
            oid=102, source=station_B.getId(), target=station_C.getId(), label='Trip',
            start_time=fixed_date, properties={'bike_type': 'classic', 'trip_data': create_trip_series(fixed_date)}
        )

        trip_A_C = self.hygraph.add_pgedge(
            oid=103, source=station_A.getId(), target=station_C.getId(), label='Trip',
            start_time=fixed_date, properties={'bike_type': 'electric', 'trip_data': create_trip_series(fixed_date)}
        )

        hour = datetime(2023, 1, 1, 8, 0, 0)  # Example timestamp

        query = HyGraphQuery(self.hygraph)
        results = (
            query
            .match_node('station', label='Station')
            .match_node('next_station', label='Station')
            .match_edge('trip', label='Trip')
            .connect('station', 'trip', 'next_station')
            .where(lambda station, trip, next_station:
                   station.get_temporal_property('bike_availability')
                   .get_value_at_timestamp(hour, 'electric_bike') >
                   station.get_static_property('capacity') * 0.7 and
                   next_station.get_temporal_property('bike_availability')
                   .get_value_at_timestamp(hour, 'electric_bike') >
                   next_station.get_static_property('capacity') * 0.7)
            .return_(
                station_names=lambda station, next_station: [station.get_static_property('name'),
                                                             next_station.get_static_property('name')],
                distance=lambda trip: trip.get_static_property('trip_data').get_value_at_timestamp(hour, 'distance'),
                avg_electric_bike_availability=lambda station, next_station: np.mean([
                    station.get_temporal_property('bike_availability').get_value_at_timestamp(hour, 'electric_bike'),
                    next_station.get_temporal_property('bike_availability').get_value_at_timestamp(hour,
                                                                                                   'electric_bike')
                ])
            )
            .execute())

        for result in results:
            # Print results for debugging or further use
            print("Path station_names:", result['station_names'])
            print("Total distance:", result['distance'])
            print("avg_electric_bike_availability:", result['avg_electric_bike_availability'])


    def test_all_edges_between_node(self):

        # Define the IDs of the nodes you want to find edges between
        source_node_id = 1
        target_node_id = 2

        # Define the time interval for the query

        start_time = datetime(2023, 12, 31)

        # Construct the query
        query = HyGraphQuery(self.hygraph) \
            .match_node(alias='source_node', node_id=source_node_id) \
            .match_node(alias='target_node', node_id=target_node_id) \
            .match_edge(alias='edge') \
            .connect('source_node', 'edge', 'target_node') \
            .where(lambda edge: pd.Timestamp(edge.start_time) >= pd.Timestamp(start_time))  \
            .return_(edges=lambda result: result['edge']).execute()

        # Print the matching edges
        for edge in query:
            print(edge)

if __name__ == '__main__':
    unittest.main()




