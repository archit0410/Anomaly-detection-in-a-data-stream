# Anomaly detection using `Isolation Forest for Streaming Anomaly Detection` (iForestASD)

Note : This project is a part of submission to Cobblestone Energy as part of their recruitment process. I am truly grateful to the recruitment team at Cobblestone Energy for this opportunity. 

## Overview

The **Isolation Forest for Streaming Anomaly Detection (iForestASD)** is a real-time system designed to identify anomalies in continuous data streams. Leveraging the **Isolation Forest** algorithm from scikit-learn, this system processes incoming data in sliding windows, detects outliers, and visualizes the results dynamically. This README provides an overview of the project, its underlying concepts, references to relevant research papers, and instructions on how to set it up and run.

Note : The `anomaly-detector.ipynb` is mainly to provide explaination to the code I have written, since Jupyter Notebooks handle plotting differently than standalone Python scripts, and certain configurations are required for the animation to work correctly in notebooks. Thus to run the code use the `anomaly-detector.py`

## Features

- **Continuous Data Stream Generation**: Simulates a data stream with regular patterns, seasonal variations, noise, and occasional anomalies.
- **Sliding Window Anomaly Detection**: Utilizes a fixed-size sliding window to maintain recent data points for model training and anomaly scoring.
- **Isolation Forest Implementation**: Employs scikit-learn's Isolation Forest for efficient unsupervised anomaly detection.
- **Drift Detection**: Monitors anomaly rates to detect concept drift and retrains the model when necessary.
- **Real-Time Visualization**: Visualizes the data stream and detected anomalies in real-time using Matplotlib's animation capabilities.

## How It Works

1. **Data Generation**:
    - The `continuous_data_stream` function generates a simulated data stream combining a regular sine wave (`regular_pattern`), a slower seasonal sine wave (`seasonal_pattern`), Gaussian noise, and occasional large spikes representing anomalies.

2. **Anomaly Detection**:
    - **Sliding Window**: Maintains a window of the latest 100 data points (`window_size`) to capture recent trends and patterns.
    - **Model Training**: Trains an Isolation Forest model on the data within the current window.
    - **Anomaly Scoring**: Scores new data points to determine their likelihood of being anomalies.
    - **Drift Detection**: Monitors the rate of detected anomalies. If the anomaly rate exceeds a predefined threshold (`drift_threshold`), the model is retrained to adapt to potential changes in data distribution.

3. **Visualization**:
    - Uses Matplotlib's `FuncAnimation` to update the plot in real-time, displaying the data stream and highlighting detected anomalies.

## Research References

The implementation of iForestASD is inspired by foundational research in anomaly detection and concept drift. Key references include:

- [An Anomaly Detection Approach Based on Isolation Forest for Streaming Data Using Sliding Window (Ding & Fei, 2013)](https://www.sciencedirect.com/science/article/pii/S1474667016314999?via%3Dihub) [1]
- [Isolation Forest (Liu, Ting, & Zhou, 2011)](https://www.researchgate.net/publication/224384174_Isolation_Forest) [2]
- [Anomaly Detection for Data Streams Based on Isolation Forest using Scikit-multiflow](https://hal.science/hal-02874869v2/file/IForestASD_ScikitMultiflow_Version2_PourHAL.pdf) [3]
- [Review of Anomaly Detection Algorithms for Data Streams](https://doi.org/10.3390/app13106353) [4]
- [Concept Drift blog post](https://neptune.ai/blog/concept-drift-best-practices) [5]
- [iForestASD code reference](https://github.com/MariamBARRY/skmultiflow_IForestASD/tree/master) [6] 

These papers provide insights into the mechanisms of anomaly detection, the effectiveness of Isolation Forests, and strategies for handling concept drift in streaming data.

## Installation and Usage

### Prerequisites

Ensure you have Python 3.6 or higher installed. The following Python packages are required:

- `numpy`
- `matplotlib`
- `scikit-learn`

You can install them using `pip`:

```bash
pip install numpy matplotlib scikit-learn
```

### Running the Code

1. **Save the Script**:

    Save the provided Python code to a file named `anomaly-detector.py`.

2. **Execute the Script**:

    Run the script using Python:

    ```bash
    python anomaly-detector.py
    ```

    A Matplotlib window will appear, displaying the real-time data stream with anomalies highlighted in red.

### Code Structure

- **Data Generation**:
    - `continuous_data_stream`: A generator function that simulates a continuous data stream with regular patterns, seasonal variations, noise, and injected anomalies.

- **Anomaly Detection Class**:
    - `iForestASD`: Encapsulates the logic for anomaly detection using Isolation Forest. It manages the sliding window, model training, anomaly scoring, drift detection, and real-time visualization.

- **Main Execution**:
    - Initializes the `iForestASD` class.
    - Creates the data stream generator.
    - Starts processing the data stream and visualizing the results.

## Customization

You can modify the behavior of the anomaly detection system by adjusting the following parameters in the `iForestASD` class:

- `window_size`: The number of recent data points to include in the sliding window.
- `n_estimators`: The number of trees in the Isolation Forest model.
- `drift_threshold`: The anomaly rate threshold that triggers model retraining.

Additionally, you can tweak the `continuous_data_stream` function parameters:

- `seasonality_period`: The period of the seasonal component in the data stream.
- `anomaly_rate`: The probability of an anomaly occurring at each step.
- `noise_level`: The standard deviation of the Gaussian noise added to the data.

## Acknowledgments

This project was developed based on insights from seminal research in anomaly detection and machine learning. Special thanks to the authors of the referenced papers for their foundational work that made this implementation possible.

## Contact

For questions, suggestions, or contributions, please contact:

- **Archit Panda**
- **Email**: [realarchit83@gmail.com]
- **GitHub**: [github-profile](https://github.com/archit0410)
