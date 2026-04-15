"""
Sensor data models for Bristlemouth-enabled devices.

This module provides sensor classes for common oceanographic measurements
compatible with the Sofar Ocean Spotter buoy and Smart Mooring system.
"""

import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .protocol import (
    BristlemouthMessage,
    BristlemouthTopic,
    DataType,
    MessageType,
    SpotterTopics,
    encode_sensor_value,
)


class SensorUnit(Enum):
    """Standard units for sensor measurements."""
    CELSIUS = "°C"
    FAHRENHEIT = "°F"
    KELVIN = "K"
    DECIBAR = "dbar"
    PASCAL = "Pa"
    BAR = "bar"
    PSU = "PSU"  # Practical Salinity Unit
    PPT = "ppt"  # Parts per thousand
    METERS_PER_SECOND = "m/s"
    CENTIMETERS_PER_SECOND = "cm/s"
    NTU = "NTU"  # Nephelometric Turbidity Unit
    FNU = "FNU"  # Formazin Nephelometric Unit
    METERS = "m"
    DEGREES = "°"
    HERTZ = "Hz"
    MILLIGRAMS_PER_LITER = "mg/L"
    PERCENT = "%"


@dataclass
class SensorReading:
    """
    A single sensor reading with value, unit, and metadata.

    Attributes:
        value: The measured value
        unit: The unit of measurement
        timestamp: Unix timestamp of the reading
        quality: Quality indicator (0-100, 100 being highest quality)
        sensor_id: Unique identifier for the sensor
        depth_m: Depth of measurement in meters (for subsurface sensors)
    """
    value: float
    unit: SensorUnit
    timestamp: float = field(default_factory=time.time)
    quality: int = 100
    sensor_id: Optional[str] = None
    depth_m: Optional[float] = None

    def __post_init__(self):
        if not 0 <= self.quality <= 100:
            raise ValueError("Quality must be between 0 and 100")

    def to_bytes(self) -> bytes:
        """Encode reading to bytes for Bristlemouth transmission."""
        # Format: value (float64) + timestamp (uint64) + quality (uint8) + depth (float32)
        depth = self.depth_m if self.depth_m is not None else -1.0
        return struct.pack('!dQBf', self.value, int(self.timestamp * 1000), self.quality, depth)

    @classmethod
    def from_bytes(cls, data: bytes, unit: SensorUnit) -> 'SensorReading':
        """Decode reading from bytes."""
        value, timestamp_ms, quality, depth = struct.unpack('!dQBf', data[:21])
        return cls(
            value=value,
            unit=unit,
            timestamp=timestamp_ms / 1000.0,
            quality=quality,
            depth_m=depth if depth >= 0 else None
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert reading to dictionary."""
        return {
            "value": self.value,
            "unit": self.unit.value,
            "timestamp": self.timestamp,
            "quality": self.quality,
            "sensor_id": self.sensor_id,
            "depth_m": self.depth_m,
        }


class BaseSensor(ABC):
    """Abstract base class for Bristlemouth-enabled sensors."""

    def __init__(
        self,
        sensor_id: str,
        topic: BristlemouthTopic,
        default_unit: SensorUnit,
        min_value: float = float('-inf'),
        max_value: float = float('inf'),
    ):
        self.sensor_id = sensor_id
        self.topic = topic
        self.default_unit = default_unit
        self.min_value = min_value
        self.max_value = max_value
        self._sequence_number = 0
        self._readings: List[SensorReading] = []

    @abstractmethod
    def validate_reading(self, value: float) -> Tuple[bool, str]:
        """Validate a sensor reading value."""
        pass

    def create_reading(
        self,
        value: float,
        unit: Optional[SensorUnit] = None,
        timestamp: Optional[float] = None,
        quality: int = 100,
        depth_m: Optional[float] = None,
    ) -> SensorReading:
        """
        Create a new sensor reading.

        Args:
            value: The measured value
            unit: Unit of measurement (defaults to sensor's default unit)
            timestamp: Unix timestamp (defaults to current time)
            quality: Quality indicator 0-100
            depth_m: Depth of measurement in meters

        Returns:
            SensorReading object

        Raises:
            ValueError: If the reading value is invalid
        """
        is_valid, msg = self.validate_reading(value)
        if not is_valid:
            raise ValueError(f"Invalid reading: {msg}")

        reading = SensorReading(
            value=value,
            unit=unit or self.default_unit,
            timestamp=timestamp or time.time(),
            quality=quality,
            sensor_id=self.sensor_id,
            depth_m=depth_m,
        )
        self._readings.append(reading)
        return reading

    def wrap_reading(self, reading: SensorReading) -> BristlemouthMessage:
        """
        Wrap a sensor reading in a Bristlemouth message.

        Args:
            reading: The sensor reading to wrap

        Returns:
            BristlemouthMessage ready for transmission
        """
        payload = reading.to_bytes()
        self._sequence_number += 1

        return BristlemouthMessage(
            topic=self.topic,
            payload=payload,
            message_type=MessageType.SENSOR_DATA,
            sequence_number=self._sequence_number,
            timestamp=reading.timestamp,
        )

    def create_and_wrap(
        self,
        value: float,
        unit: Optional[SensorUnit] = None,
        timestamp: Optional[float] = None,
        quality: int = 100,
        depth_m: Optional[float] = None,
    ) -> BristlemouthMessage:
        """
        Create a reading and wrap it in a Bristlemouth message.

        Convenience method combining create_reading and wrap_reading.
        """
        reading = self.create_reading(value, unit, timestamp, quality, depth_m)
        return self.wrap_reading(reading)

    def get_readings(self) -> List[SensorReading]:
        """Get all stored readings."""
        return self._readings.copy()

    def clear_readings(self) -> None:
        """Clear stored readings."""
        self._readings.clear()


class TemperatureSensor(BaseSensor):
    """
    Temperature sensor for oceanographic measurements.

    Compatible with Sofar Ocean's Bristlemouth-enabled temperature sensors
    for Smart Mooring deployments.
    """

    def __init__(
        self,
        sensor_id: str = "temp_sensor_01",
        min_temp_c: float = -5.0,
        max_temp_c: float = 50.0,
    ):
        super().__init__(
            sensor_id=sensor_id,
            topic=SpotterTopics.TEMPERATURE,
            default_unit=SensorUnit.CELSIUS,
            min_value=min_temp_c,
            max_value=max_temp_c,
        )

    def validate_reading(self, value: float) -> Tuple[bool, str]:
        """Validate temperature reading."""
        if not isinstance(value, (int, float)):
            return False, "Temperature must be numeric"
        if value < self.min_value:
            return False, f"Temperature {value}°C below minimum {self.min_value}°C"
        if value > self.max_value:
            return False, f"Temperature {value}°C above maximum {self.max_value}°C"
        return True, "Valid"


class PressureSensor(BaseSensor):
    """
    Pressure sensor for depth and water column measurements.

    Pressure readings can be used to calculate depth in the water column.
    """

    def __init__(
        self,
        sensor_id: str = "pressure_sensor_01",
        min_pressure_dbar: float = 0.0,
        max_pressure_dbar: float = 600.0,  # ~6000m depth
    ):
        super().__init__(
            sensor_id=sensor_id,
            topic=SpotterTopics.PRESSURE,
            default_unit=SensorUnit.DECIBAR,
            min_value=min_pressure_dbar,
            max_value=max_pressure_dbar,
        )

    def validate_reading(self, value: float) -> Tuple[bool, str]:
        """Validate pressure reading."""
        if not isinstance(value, (int, float)):
            return False, "Pressure must be numeric"
        if value < self.min_value:
            return False, f"Pressure {value} dbar below minimum {self.min_value} dbar"
        if value > self.max_value:
            return False, f"Pressure {value} dbar above maximum {self.max_value} dbar"
        return True, "Valid"

    def pressure_to_depth(self, pressure_dbar: float, latitude: float = 45.0) -> float:
        """
        Convert pressure to depth using UNESCO formula.

        Args:
            pressure_dbar: Pressure in decibars
            latitude: Latitude in degrees (affects gravity)

        Returns:
            Depth in meters
        """
        # Simplified UNESCO formula
        x = (
            (pressure_dbar * pressure_dbar * 2.2e-6)
            - (pressure_dbar * 9.9e-2)
            + 9.72659
        )
        g = 9.780318 * (1.0 + 5.2788e-3 * (latitude / 57.29578) ** 2)
        depth = pressure_dbar / (g / 10.0)
        return depth


class SalinitySensor(BaseSensor):
    """
    Salinity sensor for measuring water salinity.

    Measures salinity in Practical Salinity Units (PSU).
    """

    def __init__(
        self,
        sensor_id: str = "salinity_sensor_01",
        min_salinity_psu: float = 0.0,
        max_salinity_psu: float = 45.0,
    ):
        super().__init__(
            sensor_id=sensor_id,
            topic=SpotterTopics.SALINITY,
            default_unit=SensorUnit.PSU,
            min_value=min_salinity_psu,
            max_value=max_salinity_psu,
        )

    def validate_reading(self, value: float) -> Tuple[bool, str]:
        """Validate salinity reading."""
        if not isinstance(value, (int, float)):
            return False, "Salinity must be numeric"
        if value < self.min_value:
            return False, f"Salinity {value} PSU below minimum {self.min_value} PSU"
        if value > self.max_value:
            return False, f"Salinity {value} PSU above maximum {self.max_value} PSU"
        return True, "Valid"


class CurrentMeterSensor(BaseSensor):
    """
    Current meter sensor for measuring water current velocity.

    Measures current speed and direction for oceanographic studies.
    """

    def __init__(
        self,
        sensor_id: str = "current_sensor_01",
        max_speed_ms: float = 5.0,
    ):
        super().__init__(
            sensor_id=sensor_id,
            topic=SpotterTopics.CURRENT,
            default_unit=SensorUnit.METERS_PER_SECOND,
            min_value=0.0,
            max_value=max_speed_ms,
        )
        self._direction_readings: List[Tuple[float, float]] = []  # (speed, direction)

    def validate_reading(self, value: float) -> Tuple[bool, str]:
        """Validate current speed reading."""
        if not isinstance(value, (int, float)):
            return False, "Current speed must be numeric"
        if value < self.min_value:
            return False, f"Current speed {value} m/s cannot be negative"
        if value > self.max_value:
            return False, f"Current speed {value} m/s above maximum {self.max_value} m/s"
        return True, "Valid"

    def create_current_reading(
        self,
        speed_ms: float,
        direction_deg: float,
        timestamp: Optional[float] = None,
        quality: int = 100,
        depth_m: Optional[float] = None,
    ) -> Tuple[SensorReading, float]:
        """
        Create a current reading with speed and direction.

        Args:
            speed_ms: Current speed in meters per second
            direction_deg: Current direction in degrees (0-360, 0=North)
            timestamp: Unix timestamp
            quality: Quality indicator 0-100
            depth_m: Depth of measurement

        Returns:
            Tuple of (SensorReading for speed, direction in degrees)
        """
        if not 0 <= direction_deg <= 360:
            raise ValueError("Direction must be between 0 and 360 degrees")

        reading = self.create_reading(speed_ms, timestamp=timestamp, quality=quality, depth_m=depth_m)
        self._direction_readings.append((speed_ms, direction_deg))
        return reading, direction_deg

    def wrap_current_reading(
        self,
        reading: SensorReading,
        direction_deg: float,
    ) -> BristlemouthMessage:
        """
        Wrap a current reading with direction in a Bristlemouth message.

        The payload includes both speed and direction.
        """
        # Extended payload: reading bytes + direction (float32)
        payload = reading.to_bytes() + struct.pack('!f', direction_deg)
        self._sequence_number += 1

        return BristlemouthMessage(
            topic=self.topic,
            payload=payload,
            message_type=MessageType.SENSOR_DATA,
            sequence_number=self._sequence_number,
            timestamp=reading.timestamp,
        )


class TurbiditySensor(BaseSensor):
    """
    Turbidity sensor for measuring water clarity.

    Measures turbidity in Nephelometric Turbidity Units (NTU).
    """

    def __init__(
        self,
        sensor_id: str = "turbidity_sensor_01",
        min_turbidity_ntu: float = 0.0,
        max_turbidity_ntu: float = 4000.0,
    ):
        super().__init__(
            sensor_id=sensor_id,
            topic=SpotterTopics.TURBIDITY,
            default_unit=SensorUnit.NTU,
            min_value=min_turbidity_ntu,
            max_value=max_turbidity_ntu,
        )

    def validate_reading(self, value: float) -> Tuple[bool, str]:
        """Validate turbidity reading."""
        if not isinstance(value, (int, float)):
            return False, "Turbidity must be numeric"
        if value < self.min_value:
            return False, f"Turbidity {value} NTU cannot be negative"
        if value > self.max_value:
            return False, f"Turbidity {value} NTU above maximum {self.max_value} NTU"
        return True, "Valid"

    def classify_turbidity(self, value: float) -> str:
        """
        Classify turbidity level.

        Args:
            value: Turbidity in NTU

        Returns:
            Classification string
        """
        if value < 1:
            return "Excellent"
        elif value < 5:
            return "Good"
        elif value < 50:
            return "Fair"
        elif value < 500:
            return "Poor"
        else:
            return "Very Poor"
