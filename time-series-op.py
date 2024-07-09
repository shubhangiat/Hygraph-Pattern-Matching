import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

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
