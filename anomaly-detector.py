import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from matplotlib.animation import FuncAnimation
import random  


def continuous_data_stream(seasonality_period=200, anomaly_rate=0.03, noise_level=0.1):
    """
    Continuously generates a simulated data stream with regular patterns, seasonal elements, and random noise.
    
    Args:
        seasonality_period (int): The period of the seasonal component.
        anomaly_rate (float): Probability of an anomaly in the data.
        noise_level (float): The standard deviation of the noise.
    
    Yields:
        float: A single data point in the stream.
    """
    
    # Time index, continuously increasing
    t = 0
    
    while True:
        # Regular pattern (sine wave)
        regular_pattern = np.sin(2 * np.pi * t / seasonality_period)

        # Seasonal element (a slower sine wave, simulating seasonal change)
        seasonal_pattern = 0.5 * np.sin(2 * np.pi * t / (seasonality_period * 5))

        # Random noise
        noise = np.random.normal(0, noise_level)

        # Combine regular, seasonal, and noise components
        data_point = regular_pattern + seasonal_pattern + noise

        # Occasionally add anomalies (random spikes)
        if random.random() < anomaly_rate:
            data_point += random.uniform(5, 10)  # Inject an anomaly (a large spike)

        # Yield the data point to simulate continuous streaming
        yield data_point

        # Increment time index
        t += 1

class iForestASD:
    def __init__(self):
        """
        Initializes the IsolationForestStream.

        Parameters:
        - window_size: int, size of the sliding window.
        - n_estimators: int, number of trees in the Isolation Forest.
        - drift_threshold: float, threshold for drift detection to retrain the model.
        """

        # self is necessary for the attributes to be accessible in other methods
        self.window_size = 100 # M, size of Zi
        self.n_estimators = 25 # Number of trees in the Isolation Forest [6]
        self.drift_threshold = 0.95 # Threshold to detect drift, ideally 0.5 [6]
        self.model = None 
        self.prev_window = [] # Zi
        self.current_window = [] # Zi+1
        self.anomalies_x = []  # to store the x-coordinate of the anomalies(time)
        self.anomalies_y = []  # to store the y-coordinate of the anomalies(value)

    # E = iForest(Zi, L, N)
    def fit(self, data):
        """
        Fits the Isolation Forest model on the provided data.

        Parameters:
        - data: array-like of shape (n_samples, n_features)
        """
        self.model = IsolationForest(n_estimators=self.n_estimators, contamination=0.02, random_state=42)
        self.model.fit(data)
        print("Model trained on new window.")

    # Score = E(Zi+1)
    def score(self, data):
        """
        Scores the data using the fitted Isolation Forest model.

        Parameters:
        - data: array-like of shape (n_samples, n_features)

        Returns:
        - scores: array of shape (n_samples,), anomaly scores. The negative scores are anomalies.
        """
        return self.model.decision_function(data)

    def update_model(self, data):
        """
        Retrains the Isolation Forest model, typically after detecting drift.

        Parameters:
        - data: array-like of shape (n_samples, n_features)
        """
        print("Drift detected! Retraining model...")
        self.fit(data)

    def process_stream(self, data_stream):
        """
        Processes the continuous data stream, detects anomalies, and visualizes the results.

        Parameters:
        - data_stream: generator, the continuous data stream.
        """

        # Standard boilerplate code for plotting
        fig, ax = plt.subplots()
        xdata, ydata = [], []
        ln, = ax.plot([], [], 'b-', label='Data Stream') # ln is the data stream line
        anomaly_points, = ax.plot([], [], 'ro', label='Anomalies') # anomaly_points are the anomalies

        ax.set_xlim(0, self.window_size) 
        ax.set_ylim(-3, 13) # why -3 and 12 ?
        # max value of data stream = +1(regular) + 0.5(seasonal) + 0.3(noise) + 10(anomaly) ≈ +12.8 
        # min value of data stream = -1(regular) - 0.5(seasonal) - 0.3(noise) ≈ -1.8 (since anomaly is taken positive here no need to add it)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right')

        def update_plot(frame):
            new_data = next(data_stream)
            self.current_window.append(new_data)
            xdata.append(frame)
            ydata.append(new_data)

            # Once the current window reaches the size limit, fit and score
            if len(self.current_window) == self.window_size:
                current_window_reshaped = np.array(self.current_window).reshape(-1, 1) # Why reshape ?
                # Since scikit-learn model isolation Forest expects  expects the input to be a 2D array where: 
                #   Each row represents a sample. 
                #   Each column represents a feature.
                #   Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
                #   Reference: https://www.w3schools.com/python/numpy/numpy_array_reshape.asp

                # Will be executed only once, when the first window is processed
                if not self.model:
                    # Initial training on the first window
                    self.fit(current_window_reshaped)

                    # Score the current window based on the previously fitted model
                    scores = self.score(current_window_reshaped)

                    # Identify anomalies
                    anomalies = scores < 0
                    anomaly_indices = np.where(anomalies)[0]

                    # Record anomalies for plotting,
                    for idx in anomaly_indices:
                        anomaly_x = xdata[-self.window_size + idx]
                        anomaly_y = ydata[-self.window_size + idx]
                        self.anomalies_x.append(anomaly_x)
                        self.anomalies_y.append(anomaly_y)

                else:
                    # Score the current window based on the previously fitted model
                    scores = self.score(current_window_reshaped)

                    # Identify anomalies
                    anomalies = scores < 0  # Anomalies have negative scores
                    anomaly_indices = np.where(anomalies)[0]

                    # Record anomalies for plotting, 
                    for idx in anomaly_indices:
                        anomaly_x = xdata[-self.window_size + idx]
                        anomaly_y = ydata[-self.window_size + idx]
                        self.anomalies_x.append(anomaly_x)
                        self.anomalies_y.append(anomaly_y)

                    # Calculate anomaly rate
                    self.anomaly_rate = np.mean(anomalies)
                    print(f"Anomaly rate: {self.anomaly_rate:.2f}")

                    # If anomaly rate exceeds drift threshold, update the model
                    if self.anomaly_rate >= self.drift_threshold:
                        self.update_model(current_window_reshaped)

                # i = i+1 step
                self.prev_window = self.current_window.copy()
                self.current_window = []

            # Update plot data
            ln.set_data(xdata, ydata)
            anomaly_points.set_data(self.anomalies_x, self.anomalies_y)

            # Dynamically adjust x-axis
            if len(xdata) > self.window_size:
                ax.set_xlim(xdata[-self.window_size], xdata[-1])
            else:
                ax.set_xlim(0, self.window_size)

            return ln, anomaly_points

        # Create the animation
        ani = FuncAnimation(fig, update_plot, frames=np.arange(0, 100000), blit=True, interval=100)
        plt.show()

# Main execution
if __name__ == "__main__":
    # Initialize the stream processor with desired parameters
    stream_processor = iForestASD()
    
    # Create the data stream
    data_stream = continuous_data_stream()
    
    # Start processing the data stream
    stream_processor.process_stream(data_stream)

