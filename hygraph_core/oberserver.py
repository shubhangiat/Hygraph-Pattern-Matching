# Observer pattern
class Subject:
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer):
        try:
            self._observers.remove(observer)
        except ValueError:
            pass

    def notify(self, modifier=None):
        for observer in self._observers:
            if modifier != observer:
                observer.update(self)


import numpy as np
import xarray as xr
import pandas as pd


def dtw_i(ts1: TimeSeries, ts2: TimeSeries, window: int = None) -> float:
    """
    Independent DTW for multivariate time series

    Parameters:
    ts1, ts2: TimeSeries objects containing multivariate time series data
    window: Warping window size (optional)

    Returns:
    float: Sum of DTW distances across dimensions
    """
    total_distance = 0

    # Get variables (dimensions)
    variables = ts1.data.coords['variable'].values

    # Calculate DTW independently for each dimension
    for var in variables:
        # Extract univariate series for current dimension
        series1 = ts1.data.sel(variable=var).values
        series2 = ts2.data.sel(variable=var).values

        # Calculate DTW for this dimension
        dist = dtw_univariate(series1, series2, window)
        total_distance += dist

    return total_distance


def dtw_d(ts1: TimeSeries, ts2: TimeSeries, window: int = None) -> float:
    """
    Dependent DTW for multivariate time series

    Parameters:
    ts1, ts2: TimeSeries objects containing multivariate time series data
    window: Warping window size (optional)

    Returns:
    float: DTW distance considering all dimensions together
    """
    # Get the time series data as numpy arrays
    series1 = ts1.data.transpose('time', 'variable').values
    series2 = ts2.data.transpose('time', 'variable').values

    L1 = len(series1)
    L2 = len(series2)

    # Initialize DTW matrix
    dtw_matrix = np.full((L1 + 1, L2 + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Calculate warping window if not provided
    if window is None:
        window = max(L1, L2)

    # Fill DTW matrix
    for i in range(1, L1 + 1):
        # Calculate window bounds
        window_start = max(1, i - window)
        window_end = min(L2 + 1, i + window + 1)

        for j in range(window_start, window_end):
            # Calculate Euclidean distance between multivariate points
            dist = np.sqrt(np.sum((series1[i - 1] - series2[j - 1]) ** 2))

            # Update DTW matrix
            dtw_matrix[i, j] = dist + min(
                dtw_matrix[i - 1, j],  # insertion
                dtw_matrix[i, j - 1],  # deletion
                dtw_matrix[i - 1, j - 1]  # match
            )

    return dtw_matrix[L1, L2]


def dtw_univariate(series1: np.ndarray, series2: np.ndarray, window: int = None) -> float:
    """
    DTW for univariate time series

    Parameters:
    series1, series2: 1D numpy arrays containing univariate time series
    window: Warping window size (optional)

    Returns:
    float: DTW distance
    """
    L1 = len(series1)
    L2 = len(series2)

    # Initialize DTW matrix
    dtw_matrix = np.full((L1 + 1, L2 + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Calculate warping window if not provided
    if window is None:
        window = max(L1, L2)

    # Fill DTW matrix
    for i in range(1, L1 + 1):
        window_start = max(1, i - window)
        window_end = min(L2 + 1, i + window + 1)

        for j in range(window_start, window_end):
            dist = (series1[i - 1] - series2[j - 1]) ** 2
            dtw_matrix[i, j] = dist + min(
                dtw_matrix[i - 1, j],  # insertion
                dtw_matrix[i, j - 1],  # deletion
                dtw_matrix[i - 1, j - 1]  # match
            )

    return dtw_matrix[L1, L2]


# Example usage:
if __name__ == "__main__":
    # Create sample time series
    timestamps = pd.date_range('2024-01-01', periods=100, freq='H')
    variables = ['var1', 'var2', 'var3']

    # Sample data for ts1
    data1 = np.random.rand(100, 3)
    ts1 = TimeSeries('ts1', timestamps, variables, data1)

    # Sample data for ts2
    data2 = np.random.rand(100, 3)
    ts2 = TimeSeries('ts2', timestamps, variables, data2)

    # Calculate DTW distances
    dist_i = dtw_i(ts1, ts2, window=10)
    dist_d = dtw_d(ts1, ts2, window=10)

    print(f"Independent DTW distance: {dist_i}")
    print(f"Dependent DTW distance: {dist_d}")