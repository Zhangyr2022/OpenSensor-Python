from typing import Any, Optional
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import numpy as np  # 导入numpy库
from kalman_filter import KalmanFilter  # 请根据实际情况导入KalmanFilter类
from server import Server  # 请根据实际情况导入Server类

# 模拟 StateDimension
StateDimension = 2


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

    _X: np.ndarray  # 使用numpy数组表示向量

    @property
    def X(self):
        return self._X

    def predict(self) -> np.ndarray:
        """Predict method returns the current state."""
        return self.X

    def update(self, Z: np.ndarray) -> None:
        """Update method sets the state to the measurement."""
        self.X = Z


class Backend:
    """Backend class."""

    def __init__(self, server_port: int, filter_name: str):
        self._filter = self._build_filter(filter_name, StateDimension)
        self._logger = logging.getLogger("Backend")
        self._server = Server(
            server_port, self._handle_server_after_message_receive_event
        )

    async def start(self):
        """Starts the backend."""
        await self._server.start()

    def _handle_server_after_message_receive_event(self, sender: Optional[Any], e: str):
        """Handles the server's after message receive event."""
        self._logger.info("Received message: %s", e)

    @staticmethod
    def _build_filter(filter_name: str, initial_state_dimension: int) -> IFilter:
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
