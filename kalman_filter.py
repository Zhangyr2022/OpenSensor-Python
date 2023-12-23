from typing import Tuple
import numpy as np
from numpy.linalg import inv


class KalmanFilter:
    def __init__(self, initialX, initialP, F=None, H=None, Q=None, R=None):
        """
        KalmanFilter constructor.

        Parameters:
        - initialX: Initial state vector
        - initialP: Initial covariance matrix
        - F: State transition matrix (default is identity matrix)
        - H: Measurement matrix (default is identity matrix)
        - Q: Process noise covariance matrix (default is identity matrix)
        - R: Measurement noise covariance matrix (default is identity matrix)
        """
        self.F = F if F is not None else np.eye(len(initialX))
        self.H = H if H is not None else np.eye(len(initialX))
        self.Q = Q if Q is not None else np.eye(len(initialX)) * 5
        self.R = R if R is not None else np.eye(len(initialX)) * 3

        self.X = initialX
        self.P = initialP

        # Validate dimensions
        if self.F.shape != (len(initialX), len(initialX)):
            raise ValueError("F and X must have the same dimensions")
        if self.H.shape != (len(initialX), len(initialX)):
            raise ValueError("H and R must have compatible dimensions")
        if self.Q.shape != (len(initialX), len(initialX)):
            raise ValueError("Q must be a square matrix with the same dimensions as X")
        if self.R.shape != (len(initialX), len(initialX)):
            raise ValueError("R must be a square matrix")
        if self.P.shape != (len(initialX), len(initialX)):
            raise ValueError("P must be a square matrix with the same dimensions as X")

    def predict(self):
        """
        Predicts the next state.

        Returns:
        - Prediction vector
        """
        prediction = self._predict_dist()
        return prediction[0]

    def update(self, Z):
        """
        Updates the state based on measurement.

        Parameters:
        - Z: Measurement vector
        """
        prediction = self._predict_dist()
        predictedX, predictedP = prediction

        Y = Z - np.dot(self.H, predictedX)
        S = np.dot(np.dot(self.H, predictedP), self.H.T) + self.R
        K = np.dot(np.dot(predictedP, self.H.T), inv(S))
        updatedX = predictedX + np.dot(K, Y)
        updatedP = np.dot((np.eye(K.shape[0]) - np.dot(K, self.H)), predictedP)

        self.X = updatedX
        self.P = updatedP

    def _predict_dist(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts the next state and covariance matrix.

        Returns:
        - Tuple containing predicted state vector and covariance matrix
        """
        predictedX = np.dot(self.F, self.X)
        predictedP = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return predictedX, predictedP
