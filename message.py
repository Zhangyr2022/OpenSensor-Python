from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class Message:
    """Base class for messages."""

    message_type: str = ""

    @property
    def json(self) -> str:
        """Converts the message to a JSON string."""
        return json.dumps(self.__dict__)


@dataclass
class Gyroscope:
    """Gyroscope data class."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Accelerometer:
    """Accelerometer data class."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Gravity:
    """Gravity data class."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class SensorMessage(Message):
    """Sensor message class."""

    message_type: str = "SENSOR"
    gyroscope: Gyroscope = Gyroscope()
    accelerometer: Accelerometer = Accelerometer()
    gravity: Gravity = Gravity()
    timestamp: int = 0
    update_rate: float = 0.0
