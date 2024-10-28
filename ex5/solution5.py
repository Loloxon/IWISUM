import numpy as np

def reconstruction_errors(inputs: np.ndarray, reconstructions: np.ndarray) -> np.ndarray:
    """
    Calculate reconstruction errors between each input image and its reconstruction.

    :param inputs: Numpy array of original input images, shape (batch_size, height, width) or flattened (batch_size, n_features).
    :param reconstructions: Numpy array of reconstructed images with the same shape as inputs.
    :return: Numpy array (1D) of reconstruction errors for each input-reconstruction pair.
    """
    # Ensure inputs and reconstructions have the same shape
    assert inputs.shape == reconstructions.shape, "Shapes of inputs and reconstructions must match."

    # Calculate Mean Squared Error (MSE) between each input and its reconstruction
    if inputs.ndim == 3:  # Case where images are in 3D format [batch_size, height, width]
        errors = np.mean((inputs - reconstructions) ** 2, axis=(1, 2))
    elif inputs.ndim == 2:  # Case where images are flattened [batch_size, n_features]
        errors = np.mean((inputs - reconstructions) ** 2, axis=1)
    else:
        raise ValueError("Input arrays must be 2D or 3D.")

    return errors


def calc_threshold(reconstr_err_nominal: np.ndarray) -> float:
    """
    Calculate a threshold for anomaly detection based on nominal reconstruction errors.

    :param reconstr_err_nominal: Numpy array of reconstruction errors for nominal class examples.
    :return: A float value representing the anomaly detection threshold.
    """
    # Use the 95th percentile as the threshold
    threshold = np.percentile(reconstr_err_nominal, 95)

    return threshold


def detect(reconstr_err_all: np.ndarray, threshold: float) -> list:
    """
    Detect anomalies by comparing reconstruction errors to the threshold.

    :param reconstr_err_all: Numpy array of reconstruction errors.
    :param threshold: Anomaly-detection threshold.
    :return: List of 0s and 1s, where 1 indicates an anomaly and 0 indicates nominal.
    """
    # Detect anomalies (1) if error > threshold, otherwise nominal (0)
    anomalies = (reconstr_err_all > threshold).astype(int)

    return anomalies.tolist()
