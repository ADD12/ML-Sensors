"""
Bristlemouth Protocol Wrapper for Sofar Ocean Spotter Buoy.

This module provides Python utilities for wrapping sensor data using the
Bristlemouth open ocean connectivity standard for use with Sofar Ocean's
Spotter buoy platform.

Bristlemouth is an open standard delivering plug-and-play hardware interfaces
for marine applications. Learn more at https://bristlemouth.org/
"""

from .protocol import (
    BristlemouthMessage,
    BristlemouthTopic,
    MessageType,
    BCMP_PROTOCOL_VERSION,
)
from .sensors import (
    SensorReading,
    TemperatureSensor,
    PressureSensor,
    SalinitySensor,
    CurrentMeterSensor,
    TurbiditySensor,
)
from .spotter import SpotterBuoyClient, TransmissionMode

__all__ = [
    "BristlemouthMessage",
    "BristlemouthTopic",
    "MessageType",
    "BCMP_PROTOCOL_VERSION",
    "SensorReading",
    "TemperatureSensor",
    "PressureSensor",
    "SalinitySensor",
    "CurrentMeterSensor",
    "TurbiditySensor",
    "SpotterBuoyClient",
    "TransmissionMode",
]
