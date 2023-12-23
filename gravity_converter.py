import numpy as np


class GravityConverter:
    def __init__(self, initial_gravity_direction: np.ndarray):
        """
        Initializes the GravityConverter.

        Parameters:
        - initial_gravity_direction: The initial gravity direction in the device's local coordinate system.
        """
        self._initial_gravity_direction = initial_gravity_direction / np.linalg.norm(
            initial_gravity_direction
        )
        self._current_rotation_matrix = self._initialize_rotation_matrix()
        self._current_rotation_angle = 0.0

    def _initialize_rotation_matrix(self):
        """
        Initializes the rotation matrix based on the initial gravity direction.
        """
        gravity_direction = np.array(
            [0, 0, 1]
        )  # Gravity direction in the world coordinate system
        rotation_axis = np.cross(self._initial_gravity_direction, gravity_direction)
        rotation_angle = np.arccos(
            np.dot(self._initial_gravity_direction, gravity_direction)
        )
        rotation_matrix = self._rotation_matrix_from_axis_angle(
            rotation_axis, rotation_angle
        )
        return rotation_matrix

    def _rotation_matrix_from_axis_angle(self, axis, angle):
        """
        Computes the rotation matrix from an axis and an angle.

        Parameters:
        - axis: The rotation axis.
        - angle: The rotation angle (in radians).

        Returns:
        - The rotation matrix.
        """
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c

        x, y, z = axis
        rotation_matrix = np.array(
            [
                [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
            ]
        )

        return rotation_matrix

    def rotate_gravity(self, accelerometer_data):
        """
        Rotates the accelerometer data to the world coordinate system and subtracts gravity.

        Parameters:
        - accelerometer_data: The accelerometer data in the device's local coordinate system.

        Returns:
        - The rotated and gravity-subtracted accelerometer data in the world coordinate system.
        """
        rotated_acceleration = np.dot(self._current_rotation_matrix, accelerometer_data)
        # gravity_subtracted_acceleration = rotated_acceleration - np.array([0, 0, 9.81])
        return rotated_acceleration

    def update_rotation(self, gyroscope_data, time_interval):
        """
        Updates the current rotation state based on gyroscope data.

        Parameters:
        - gyroscope_data: The gyroscope data (angular velocity) in the device's local coordinate system.
        - time_interval: The time interval between updates.
        """
        gyro_magnitude = np.linalg.norm(gyroscope_data)
        if gyro_magnitude > 1e-6:
            gyro_axis = gyroscope_data / gyro_magnitude
            rotation_increment = gyro_axis * gyro_magnitude * time_interval
            rotation_matrix_increment = self._rotation_matrix_from_axis_angle(
                gyro_axis, gyro_magnitude * time_interval
            )
            self._current_rotation_matrix = np.dot(
                rotation_matrix_increment, self._current_rotation_matrix
            )
            self._current_rotation_angle += gyro_magnitude * time_interval
