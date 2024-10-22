import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import xarray as xr
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class TimeSeriesMetadata:
    def __init__(self, owner_id, element_type=''):
        self.owner_id = owner_id
        self.element_type = element_type

class TimeSeries:
    """
     Create and add a multivariate time series to the graph.
     :param tsid: Time series ID
     :param timestamps: List of timestamps
     :param variables: List of variable names
     :param data: 2D array-like structure with data
     """
    def __init__(self, tsid, timestamps, variables, data, metadata=None):
        self.tsid = tsid
        time_index = pd.to_datetime(timestamps)
        self.data = xr.DataArray(data, coords=[time_index, variables], dims=['time', 'variable'], name=f'ts_{tsid}')
        self.metadata = metadata if metadata is not None else {}

    def append_data(self, date, value):
        date = pd.to_datetime(date)
        new_data = xr.DataArray([[value]], coords=[[date], self.data.coords['variable']], dims=['time', 'variable'])
        self.data = xr.concat([self.data, new_data], dim='time')

    def get_id(self):
        return

    def apply_aggregation(self, aggregation_name):
        if aggregation_name == 'sum':
            return self.data.sum().values
        elif aggregation_name == 'mean':
            return self.data.mean().values
        elif aggregation_name == 'min':
            return self.data.min().values
        elif aggregation_name == 'max':
            return self.data.max().values
        elif aggregation_name == 'count':
            return len(self.data.coords['time'].values)
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation_name}")

    def sum(self):
        """
        Return the sum of the time series data.
        """
        return np.sum(self.data.values)

    def mean(self):
        """
        Return the mean of the time series data.
        """
        return np.mean(self.data.values)

    def max(self):
        """
        Return the maximum value of the time series data.
        """
        return np.max(self.data.values)

    def min(self):
        """
        Return the minimum value of the time series data.
        """
        return np.min(self.data.values)

    def variance(self):
        """
        Return the variance of the time series data.
        """
        return np.var(self.data.values)

    def count(self):
        """
        Return the count of data points in the time series.
        """
        return self.data.size

        # --- Access Specific Values or Timestamps ---

    def get_value_at_timestamp(self, timestamp):
        """
        Get the value(s) at a specific timestamp.
        """
        try:
            timestamp = pd.to_datetime(timestamp)
            return self.data.sel(time=timestamp).values
        except KeyError:
            raise ValueError(f"Timestamp {timestamp} not found in time series.")

    def get_timestamp_at_value(self, value):
        """
        Get the timestamp(s) where the value(s) is equal to the specified value.
        """
        matching_times = self.data.where(self.data == value, drop=True).coords['time']
        if matching_times.size == 0:
            raise ValueError(f"No matching timestamp found for value {value}.")
        return matching_times.values

    def last_value(self):
        """
        Get the last value and its timestamp from the time series.
        """
        last_timestamp = self.data.time[-1].values
        last_value = self.data.sel(time=last_timestamp).values
        return last_timestamp, last_value

    def first_value(self):
        """
        Get the first value and its timestamp from the time series.
        """
        first_timestamp = self.data.time[0].values
        first_value = self.data.sel(time=first_timestamp).values
        return first_timestamp, first_value

    def limit(self, num_points, order='last'):
        """
        Limit the time series to a specific number of points.

        Parameters:
        - num_points (int): Number of points to limit the data to.
        - order (str): Whether to get the 'first' or 'last' points. Default is 'last'.

        Returns:
        - A subset of the time series data limited by the number of points.
        """
        if order == 'first':
            limited_data = self.data.isel(time=slice(0, num_points))
        elif order == 'last':
            limited_data = self.data.isel(time=slice(-num_points, None))
        else:
            raise ValueError("Order must be either 'first' or 'last'.")
        return limited_data

    def last_timestamp(self):
        """
        Returns the last (most recent) timestamp in the time series.
        """
        return self.data.coords['time'].values[-1]

    def first_timestamp(self):
        """
        Returns the first (earliest) timestamp in the time series.
        """
        return self.data.coords['time'].values[0]

    def display_time_series(self, ts, limit=10, order='first'):
        """
               Display the time series data with optional limits on the number of data points.

               Parameters:
               - limit (int, optional): The maximum number of data points to display.
               - order (str, optional): Whether to retrieve 'first' or 'last' N data points.
               """
        print(f"Time Series {ts.tsid}: {ts.metadata.owner_id}")
        variables = [str(var) for var in ts.data.coords['variable'].values]
        print(f"Variables: {', '.join(variables)}")

        ts_df = ts.data.to_dataframe('value').reset_index()
        if limit is not None:
            if order == 'first':
                ts_df = ts_df.head(limit)
            elif order == 'last':
                ts_df = ts_df.tail(limit)
        grouped = ts_df.groupby('time')

        output = []
        for time, group in grouped:
            values = [f"{row['value']}" for idx, row in group.iterrows()]
            row_str = ", ".join(values)
            output.append(f"{time}, {row_str}")
            print (output)
    # --- Time Series Similarity ---
    def euclidean_distance(self, other_timeseries):
        """
        Compute the Euclidean distance between this time series and another.
        Both time series must have the same length.
        """
        self_values = self.data.values
        other_values = other_timeseries.data.values
        if self_values.shape != other_values.shape:
            raise ValueError("Both time series must have the same length for Euclidean distance.")
        return np.linalg.norm(self_values - other_values)

    def dynamic_time_warping(self, other_timeseries):
        """
        Compute the Dynamic Time Warping (DTW) distance between two time series.
        """
        self_df = self.to_dataframe()
        other_df = other_timeseries.to_dataframe()
        distance, _ = fastdtw(self_df['value'].values, other_df['value'].values, dist=euclidean)
        return distance

    # --- Time Series Classification ---
    def classify(self, train_data, train_labels, method='knn', **kwargs):
        """
        Train a classifier on time series data and classify this time series.
        Parameters:
        - train_data: List of time series data (each as a pandas DataFrame).
        - train_labels: List of labels corresponding to each training time series.
        - method: Classifier method ('knn', 'random_forest', 'svm').
        - kwargs: Additional arguments to pass to the classifier.
        """
        # Convert each time series to a flattened numpy array for classification
        train_data_flat = [ts['value'].values.flatten() for ts in train_data]
        test_data = self.to_dataframe()['value'].values.flatten().reshape(1, -1)

        # Select the classifier
        if method == 'knn':
            classifier = KNeighborsClassifier(**kwargs)
        elif method == 'random_forest':
            classifier = RandomForestClassifier(**kwargs)
        elif method == 'svm':
            classifier = SVC(**kwargs)
        else:
            raise ValueError(f"Unsupported classification method: {method}")

        # Train the classifier
        classifier.fit(train_data_flat, train_labels)

        # Predict the label for the current time series
        return classifier.predict(test_data)

    # --- Helper Functions ---
    def to_dataframe(self):
        """
        Convert the time-series to a pandas DataFrame.
        """
        return self.data.to_dataframe('value').reset_index()

    def autocorrelation(self, lag=1):
        """
        Compute the autocorrelation for a given lag.
        """
        df = self.to_dataframe().set_index('time')
        return df['value'].autocorr(lag=lag)

    def check_stationarity(self):
        """
        Perform the Augmented Dickey-Fuller test to check for stationarity.
        Returns the p-value of the test.
        """
        df = self.to_dataframe()
        result = adfuller(df['value'].values)
        return result[1]  # p-value

    def exponential_smoothing(self, trend=None, seasonal=None, seasonal_periods=None):
        """
        Apply Holt-Winters Exponential Smoothing.
        """
        df = self.to_dataframe().set_index('time')
        model = ExponentialSmoothing(df['value'], trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
        fitted_model = model.fit()
        return fitted_model.fittedvalues


def generate_time_series(length, noise_factor=0.1):
    """
    Generate a sine wave time series with added noise.
    :param length: Length of the time series.
    :param noise_factor: Amplitude of the noise to add.
    :return: Generated time series.
    """
    t = np.linspace(0, 4 * np.pi, length)
    series = np.sin(t) + noise_factor * np.random.randn(length)
    return series

def plot_time_series(ts1, ts2, title):
    """
    Plot two time series on the same graph.
    :param ts1: First time series.
    :param ts2: Second time series.
    :param title: Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(ts1, label='Time Series 1')
    plt.plot(ts2, label='Time Series 2')
    plt.title(title)
    plt.legend()
    plt.show()

def compute_dtw(ts1, ts2):
    """
    Compute the DTW distance between two time series.
    :param ts1: First time series.
    :param ts2: Second time series.
    :return: DTW distance and the alignment path.
    """
    distance, path = fastdtw(ts1, ts2, dist=euclidean)
    return distance, path

def main():
    # Generate similar time series
    ts1 = generate_time_series(100)
    ts2 = generate_time_series(100, noise_factor=0.1)

    # Plot similar time series
    plot_time_series(ts1, ts2, "Similar Time Series")

    # Compute DTW distance for similar time series
    distance, path = compute_dtw(ts1, ts2)
    print(f"DTW distance for similar time series: {distance}")

    # Generate different time series
    ts3 = generate_time_series(100)
    ts4 = generate_time_series(100, noise_factor=0.5) + 2  # Adding an offset to make them different

    # Plot different time series
    plot_time_series(ts3, ts4, "Different Time Series")

    # Compute DTW distance for different time series
    distance, path = compute_dtw(ts3, ts4)
    print(f"DTW distance for different time series: {distance}")

if __name__ == "__main__":
    main()
