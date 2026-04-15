"""
Sofar Ocean Spotter Buoy Client.

This module provides a client interface for interacting with the Sofar Ocean
Spotter buoy platform using the Bristlemouth protocol.
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from collections import deque

from .protocol import (
    BristlemouthMessage,
    BristlemouthTopic,
    MessageType,
    SpotterTopics,
    MAX_PAYLOAD_SIZE,
)
from .sensors import BaseSensor, SensorReading


class TransmissionMode(Enum):
    """Data transmission mode for Spotter buoy."""
    SATELLITE = "satellite"  # Iridium satellite transmission
    CELLULAR = "cellular"    # Cellular network transmission
    HYBRID = "hybrid"        # Use cellular when available, fallback to satellite
    LOCAL = "local"          # Local storage only (no transmission)


class ConnectionState(Enum):
    """Connection state for Spotter buoy."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class SpotterConfig:
    """Configuration for Spotter buoy client."""
    device_id: str
    transmission_mode: TransmissionMode = TransmissionMode.HYBRID
    sample_interval_seconds: int = 60
    transmit_interval_seconds: int = 3600  # 1 hour default
    max_queue_size: int = 1000
    enable_compression: bool = True
    api_endpoint: str = "https://api.sofarocean.com/api"


@dataclass
class TransmissionResult:
    """Result of a data transmission attempt."""
    success: bool
    message_count: int
    bytes_transmitted: int
    timestamp: float = field(default_factory=time.time)
    error_message: Optional[str] = None
    transmission_mode: Optional[TransmissionMode] = None


class SpotterBuoyClient:
    """
    Client for Sofar Ocean Spotter buoy platform.

    This client manages sensor data collection, Bristlemouth message formatting,
    and data transmission to the Sofar Ocean cloud platform.

    Example usage:
        ```python
        from ml_sensors.bristlemouth import SpotterBuoyClient, TemperatureSensor

        # Create client
        client = SpotterBuoyClient(device_id="SPOT-1234")

        # Register sensors
        temp_sensor = TemperatureSensor()
        client.register_sensor(temp_sensor)

        # Add readings
        reading = temp_sensor.create_reading(22.5)
        client.queue_reading(temp_sensor, reading)

        # Transmit data
        result = client.transmit()
        ```
    """

    def __init__(
        self,
        device_id: str,
        config: Optional[SpotterConfig] = None,
    ):
        """
        Initialize Spotter buoy client.

        Args:
            device_id: Unique identifier for the Spotter device
            config: Optional configuration (uses defaults if not provided)
        """
        self.config = config or SpotterConfig(device_id=device_id)
        self.device_id = device_id
        self._sensors: Dict[str, BaseSensor] = {}
        self._message_queue: deque = deque(maxlen=self.config.max_queue_size)
        self._transmission_history: List[TransmissionResult] = []
        self._state = ConnectionState.DISCONNECTED
        self._sequence_number = 0
        self._callbacks: Dict[str, List[Callable]] = {
            "on_transmit": [],
            "on_error": [],
            "on_state_change": [],
        }

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @state.setter
    def state(self, new_state: ConnectionState):
        """Set connection state and trigger callbacks."""
        old_state = self._state
        self._state = new_state
        if old_state != new_state:
            for callback in self._callbacks["on_state_change"]:
                callback(old_state, new_state)

    def register_sensor(self, sensor: BaseSensor) -> None:
        """
        Register a sensor with the client.

        Args:
            sensor: Sensor instance to register
        """
        self._sensors[sensor.sensor_id] = sensor

    def unregister_sensor(self, sensor_id: str) -> bool:
        """
        Unregister a sensor from the client.

        Args:
            sensor_id: ID of sensor to unregister

        Returns:
            True if sensor was found and removed, False otherwise
        """
        if sensor_id in self._sensors:
            del self._sensors[sensor_id]
            return True
        return False

    def get_sensor(self, sensor_id: str) -> Optional[BaseSensor]:
        """Get a registered sensor by ID."""
        return self._sensors.get(sensor_id)

    def list_sensors(self) -> List[str]:
        """List all registered sensor IDs."""
        return list(self._sensors.keys())

    def queue_reading(
        self,
        sensor: BaseSensor,
        reading: SensorReading,
    ) -> BristlemouthMessage:
        """
        Queue a sensor reading for transmission.

        Args:
            sensor: The sensor that produced the reading
            reading: The sensor reading to queue

        Returns:
            The Bristlemouth message that was queued
        """
        message = sensor.wrap_reading(reading)
        self._message_queue.append(message)
        return message

    def queue_message(self, message: BristlemouthMessage) -> None:
        """
        Queue a pre-formatted Bristlemouth message.

        Args:
            message: Message to queue for transmission
        """
        self._message_queue.append(message)

    def queue_raw_data(
        self,
        topic: BristlemouthTopic,
        data: bytes,
        message_type: MessageType = MessageType.SENSOR_DATA,
    ) -> BristlemouthMessage:
        """
        Queue raw data as a Bristlemouth message.

        Args:
            topic: Topic for the message
            data: Raw payload data
            message_type: Type of message

        Returns:
            The created and queued message
        """
        self._sequence_number += 1
        message = BristlemouthMessage(
            topic=topic,
            payload=data,
            message_type=message_type,
            sequence_number=self._sequence_number,
        )
        self._message_queue.append(message)
        return message

    def get_queue_size(self) -> int:
        """Get number of messages in the transmission queue."""
        return len(self._message_queue)

    def clear_queue(self) -> int:
        """
        Clear the transmission queue.

        Returns:
            Number of messages that were cleared
        """
        count = len(self._message_queue)
        self._message_queue.clear()
        return count

    def peek_queue(self, count: int = 10) -> List[BristlemouthMessage]:
        """
        Peek at messages in the queue without removing them.

        Args:
            count: Maximum number of messages to return

        Returns:
            List of messages from the front of the queue
        """
        return list(self._message_queue)[:count]

    def transmit(
        self,
        mode: Optional[TransmissionMode] = None,
        max_messages: Optional[int] = None,
    ) -> TransmissionResult:
        """
        Transmit queued messages to the Sofar Ocean platform.

        This is a simulation of the transmission process. In a real deployment,
        this would interface with the Spotter's satellite/cellular modem.

        Args:
            mode: Transmission mode to use (defaults to config setting)
            max_messages: Maximum messages to transmit (None = all)

        Returns:
            TransmissionResult with details of the transmission
        """
        mode = mode or self.config.transmission_mode

        if not self._message_queue:
            return TransmissionResult(
                success=True,
                message_count=0,
                bytes_transmitted=0,
                transmission_mode=mode,
            )

        # Determine how many messages to send
        messages_to_send = max_messages or len(self._message_queue)
        messages_to_send = min(messages_to_send, len(self._message_queue))

        # Collect messages and calculate total bytes
        transmitted_messages = []
        total_bytes = 0

        for _ in range(messages_to_send):
            if self._message_queue:
                msg = self._message_queue.popleft()
                transmitted_messages.append(msg)
                total_bytes += len(msg.to_bytes())

        # Simulate transmission (in real implementation, this would send to modem)
        result = TransmissionResult(
            success=True,
            message_count=len(transmitted_messages),
            bytes_transmitted=total_bytes,
            transmission_mode=mode,
        )

        self._transmission_history.append(result)

        # Trigger callbacks
        for callback in self._callbacks["on_transmit"]:
            callback(result)

        return result

    def get_transmission_history(
        self,
        limit: Optional[int] = None,
    ) -> List[TransmissionResult]:
        """
        Get transmission history.

        Args:
            limit: Maximum number of results to return (None = all)

        Returns:
            List of TransmissionResult objects
        """
        if limit:
            return self._transmission_history[-limit:]
        return self._transmission_history.copy()

    def create_status_message(self) -> BristlemouthMessage:
        """
        Create a status message for the Spotter buoy.

        Returns:
            BristlemouthMessage containing device status
        """
        status = {
            "device_id": self.device_id,
            "state": self._state.value,
            "queue_size": len(self._message_queue),
            "registered_sensors": list(self._sensors.keys()),
            "transmission_mode": self.config.transmission_mode.value,
            "timestamp": time.time(),
        }

        payload = json.dumps(status).encode('utf-8')
        self._sequence_number += 1

        return BristlemouthMessage(
            topic=SpotterTopics.STATUS,
            payload=payload,
            message_type=MessageType.SENSOR_DATA,
            sequence_number=self._sequence_number,
        )

    def on_transmit(self, callback: Callable[[TransmissionResult], None]) -> None:
        """Register a callback for transmission events."""
        self._callbacks["on_transmit"].append(callback)

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register a callback for error events."""
        self._callbacks["on_error"].append(callback)

    def on_state_change(
        self,
        callback: Callable[[ConnectionState, ConnectionState], None],
    ) -> None:
        """Register a callback for state change events."""
        self._callbacks["on_state_change"].append(callback)

    def format_for_api(self, messages: List[BristlemouthMessage]) -> Dict[str, Any]:
        """
        Format messages for the Sofar Ocean API.

        Args:
            messages: List of messages to format

        Returns:
            Dictionary formatted for API submission
        """
        return {
            "device_id": self.device_id,
            "timestamp": time.time(),
            "message_count": len(messages),
            "messages": [
                {
                    "topic": str(msg.topic),
                    "type": msg.message_type.name,
                    "sequence": msg.sequence_number,
                    "timestamp": msg.timestamp,
                    "payload_hex": msg.payload.hex(),
                }
                for msg in messages
            ],
        }

    def __repr__(self) -> str:
        return (
            f"SpotterBuoyClient(device_id='{self.device_id}', "
            f"state={self._state.value}, "
            f"sensors={len(self._sensors)}, "
            f"queue_size={len(self._message_queue)})"
        )


def create_spotter_client(
    device_id: str,
    transmission_mode: TransmissionMode = TransmissionMode.HYBRID,
    sample_interval: int = 60,
) -> SpotterBuoyClient:
    """
    Factory function to create a configured Spotter buoy client.

    Args:
        device_id: Unique device identifier
        transmission_mode: Mode for data transmission
        sample_interval: Sampling interval in seconds

    Returns:
        Configured SpotterBuoyClient instance
    """
    config = SpotterConfig(
        device_id=device_id,
        transmission_mode=transmission_mode,
        sample_interval_seconds=sample_interval,
    )
    return SpotterBuoyClient(device_id=device_id, config=config)
