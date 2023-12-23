from typing import Any, Optional
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import numpy as np  # 导入numpy库
from kalman_filter import KalmanFilter  # 请根据实际情况导入KalmanFilter类
from server import Server  # 请根据实际情况导入Server类
from message import (
    SensorMessage,
    Gyroscope,
    Accelerometer,
    Gravity,
)  # 请根据实际情况导入SensorMessage类
import matplotlib.pyplot as plt
from gravity_converter import GravityConverter
import os


class IFilter(ABC):
    """Interface for filters."""

    @property
    @abstractmethod
    def X(self) -> Any:
        """State vector property."""
        pass

    @abstractmethod
    def predict(self) -> Any:
        """Predict method."""
        pass

    @abstractmethod
    def update(self, Z: Any) -> None:
        """Update method."""
        pass


@dataclass
class NoFilter(IFilter):
    """NoFilter class."""

    def __init__(self):
        self._X = None

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = value

    def predict(self) -> np.ndarray:
        """Predict method returns the current state."""
        return self.X

    def update(self, Z: np.ndarray) -> None:
        """Update method sets the state to the measurement."""
        self.X = Z


class Backend:
    """Backend class."""

    async def start(self):
        """Starts the backend."""
        await self._server.start()

    def __init__(
        self,
        server_port: int,
        filter_name: str,
        state_dimension: int,
        buffer_size: int,
        real_time_plot: bool = False,
        csv_path: str = None,
    ):
        self.max_points = 160
        self._angular_velocity_filter = self._build_filter(filter_name, state_dimension)
        self._accelerometer_filter = self._build_filter(filter_name, state_dimension)
        self._velocity_filter = self._build_filter(filter_name, state_dimension)
        self._displacement_filter = self._build_filter(filter_name, state_dimension)

        self._logger = logging.getLogger("Backend")
        self._server = Server(
            server_port, self._handle_server_after_message_receive_event
        )
        self._buffer_size = buffer_size
        self._world_velocity_buffer = [np.array([0, 0, 0])]
        self._world_displacement_buffer = [np.array([0, 0, 0])]
        self._world_accel_buffer = []

        self._gyro_buffer = []
        self._accl_buffer = []
        self._grav_buffer = []
        self._time_buffer = []

        self._world_displacement_history = []  # 用于保存历史轨迹数据
        self._world_accel_history = []
        self._last_drawn_displacement = np.zeros(state_dimension)  # 用于保存上一次绘制的最后位移点

        self._fig = plt.figure()
        self._displacement_ax = self.init_axis(121, "Real-time Trajectory")
        self._a_ax = self.init_axis(122, "Real-time Acceleration")

        # self._displacement_scatter_list = []
        # self._a_scatter_list = []
        # self._removed_displacement_group_num = 0
        # self._removed_a_group_num = 0
        self._record_cnt = 0
        self.MAX_RECORD = 10000

        self._grivity_converter = None

        self._real_time_plot = real_time_plot
        self._csv_path = csv_path

        self._finish_record = False

        # 检查csv文件是否存在，如果存在则删除
        if self._csv_path is not None and os.path.exists(self._csv_path):
            os.remove(self._csv_path)
        # 在csv文件中写入表头
        if self._real_time_plot:
            with open(self._csv_path, "w") as f:
                f.write(
                    "timestamp,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z,gravity_x,gravity_y,gravity_z\n"
                )

    def init_axis(self, subplot_num, title):
        ax = self._fig.add_subplot(subplot_num, projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)

        return ax

    def _handle_server_after_message_receive_event(
        self, sender: Optional[Any], e: SensorMessage
    ):
        """Handles the server's after message receive event."""

        if self._real_time_plot:

            # 解析消息，获取陀螺仪数据
            gyroscope_data = Gyroscope(**e.gyroscope)

            gyro_measurements = np.array(
                [gyroscope_data.x, gyroscope_data.y, gyroscope_data.z]
            )

            # 调用卡尔曼滤波器的update方法，更新状态
            self._angular_velocity_filter.update(gyro_measurements)

            # 解析消息，获取线加速度和角速度数据
            accelerometer_data = Accelerometer(**e.accelerometer)
            acceleration = np.array(
                [accelerometer_data.x, accelerometer_data.y, accelerometer_data.z]
            )
            self._accelerometer_filter.update(acceleration)

            # 如果重力转换器未初始化，则初始化
            if self._grivity_converter is None:
                # 解析消息，获取线加速度和角速度数据
                gravity_data = Gravity(**e.gravity)
                self._grivity_converter = GravityConverter(
                    np.array([gravity_data.x, gravity_data.y, gravity_data.z])
                )
            else:
                # 更新旋转矩阵
                self._grivity_converter.update_rotation(
                    gyro_measurements, e.update_rate
                )

            filtered_acceleration = self._accelerometer_filter.X

            # 计算世界坐标系下的加速度
            world_acceleration = self._grivity_converter.rotate_gravity(
                filtered_acceleration
            )

            # 将数据添加到缓冲区
            self._gyro_buffer.append(gyro_measurements)
            self._world_accel_buffer.append(world_acceleration)

            new_velocity = (
                self._world_velocity_buffer[-1] + world_acceleration * e.update_rate
            )
            self._velocity_filter.update(new_velocity)

            self._world_velocity_buffer.append(self._velocity_filter.X)

            new_displacement = (
                self._world_displacement_buffer[-1]
                + self._world_velocity_buffer[-1] * e.update_rate
            )
            self._displacement_filter.update(new_displacement)

            self._world_displacement_buffer.append(self._displacement_filter.X)

            # 判断是否达到缓冲区大小
            if len(self._gyro_buffer) >= self._buffer_size:
                # 进行位移估计，这里简单地使用加速度进行积分
                world_displacement = np.array(self._world_displacement_buffer)

                # 保存轨迹数据到历史记录
                self._world_displacement_history.append(world_displacement)
                self._world_accel_history.append(np.array(self._world_accel_buffer))

                # 更新实时展示
                self._visualize_trajectory()

                self._last_drawn_displacement = world_displacement[-1]  # 保存最后位移点
                # 清空缓冲区
                self._gyro_buffer = []
                self._world_accel_buffer = []
                self._world_velocity_buffer = [self._world_velocity_buffer[-1]]
                self._world_displacement_buffer = [self._world_displacement_buffer[-1]]
        else:
            if not self._finish_record:
                if len(self._gyro_buffer) == 0:
                    self._logger.info("Start recording...")
                elif len(self._gyro_buffer) % ((self.MAX_RECORD) // 10) == 0:
                    self._logger.info(f"Record {len(self._gyro_buffer)} data...")

                # 解析消息，获取数据，并记录到csv文件中
                gyroscope_data = Gyroscope(**e.gyroscope)
                accelerometer_data = Accelerometer(**e.accelerometer)
                gravity_data = Gravity(**e.gravity)

                self._gyro_buffer.append(gyroscope_data)
                self._accl_buffer.append(accelerometer_data)
                self._grav_buffer.append(gravity_data)
                self._time_buffer.append(e.timestamp)

                if (
                    len(self._gyro_buffer) >= self.MAX_RECORD
                    and not self._finish_record
                ):
                    self._finish_record = True
                    self._logger.info("Record finished!")
                    self._logger.info(f"Saving data to csv file {self._csv_path }...")
                    with open(self._csv_path, "a") as f:
                        for gyroscope, accelerometer, gravity, timestamp in zip(
                            self._gyro_buffer,
                            self._accl_buffer,
                            self._grav_buffer,
                            self._time_buffer,
                        ):
                            f.write(
                                f"{timestamp},"
                                f"{gyroscope_data.x},{gyroscope_data.y},{gyroscope_data.z},"
                                f"{accelerometer_data.x},{accelerometer_data.y},{accelerometer_data.z},"
                                f"{gravity_data.x},{gravity_data.y},{gravity_data.z}\n"
                            )

                    # 记录数据

    def _visualize_trajectory(self):
        """Visualize the trajectory in real-time using Matplotlib."""
        groups_cnt = len(self._world_displacement_history)
        points_cnt = len(self._world_displacement_history[0]) if groups_cnt > 0 else 0
        if groups_cnt > 0 and self.max_points < groups_cnt * points_cnt:
            # 重新绘制轨迹
            old_displacement = self._world_displacement_history.pop(0)
            old_accel = self._world_accel_history.pop(0)

            self._displacement_ax.cla()
            self._a_ax.cla()

            # Set up new scatter plots
            for displacement, accel in zip(
                self._world_displacement_history, self._world_accel_history
            ):
                self._displacement_ax.scatter(
                    displacement[:, 0],
                    displacement[:, 1],
                    displacement[:, 2],
                )

                self._a_ax.scatter(
                    accel[:, 0],
                    accel[:, 1],
                    accel[:, 2],
                )

            # a_offset = self._a_scatter_list[self._removed_a_group_num].get_offsets()
            # displacement_offset = self._displacement_scatter_list[
            #     self._removed_displacement_group_num
            # ].get_offsets()
            # # a_offset = None
            # # displacement_offset = None
            # for i in range(len(a_offset)):
            #     a_offset[i] = (np.nan, np.nan)
            # for i in range(len(displacement_offset)):
            #     displacement_offset[i] = (np.nan, np.nan)
            # self._removed_a_group_num += 1
            # self._removed_displacement_group_num += 1

            # # Remove the oldest scatter plots
            # old_displacement = self._world_displacement_history.pop(0)
            # old_accel = self._world_accel_history.pop(0)

            # # Clear the axes
            # self._displacement_ax.cla()
            # self._a_ax.cla()

        else:
            # Plot the newest points
            displacement = self._world_displacement_history[-1]
            accel = self._world_accel_history[-1]

            # self._displacement_scatter_list.append(
            self._displacement_ax.scatter(
                displacement[:, 0],
                displacement[:, 1],
                displacement[:, 2],
            )
            # )

            # self._a_scatter_list.append(
            self._a_ax.scatter(
                accel[:, 0],
                accel[:, 1],
                accel[:, 2],
            )
            # )

        # Update labels and display the trajectory, non-blocking mode
        plt.draw()
        plt.pause(0.001)

    @staticmethod
    def _build_filter(filter_name: str, initial_state_dimension: int):
        """Builds a filter based on the given filter name."""
        if filter_name == "NoFilter":
            return NoFilter(np.zeros(initial_state_dimension))  # 使用numpy创建零向量

        elif filter_name == "KalmanFilter":
            return KalmanFilter(
                np.zeros(initial_state_dimension),  # 使用numpy创建零向量
                np.eye(initial_state_dimension),  # 使用numpy创建单位矩阵
            )

        else:
            raise ValueError("Invalid filter name")


# Rest of the code remains unchanged
