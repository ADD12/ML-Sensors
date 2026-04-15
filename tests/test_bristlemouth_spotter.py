"""Tests for Spotter buoy client."""

import pytest
import json
import time

from ml_sensors.bristlemouth.spotter import (
    SpotterBuoyClient,
    SpotterConfig,
    TransmissionMode,
    TransmissionResult,
    ConnectionState,
    create_spotter_client,
)
from ml_sensors.bristlemouth.sensors import (
    TemperatureSensor,
    PressureSensor,
    SensorReading,
    SensorUnit,
)
from ml_sensors.bristlemouth.protocol import (
    BristlemouthMessage,
    BristlemouthTopic,
    MessageType,
)


class TestSpotterConfig:
    """Tests for SpotterConfig class."""

    def test_create_config(self):
        """Test creating a config."""
        config = SpotterConfig(device_id="SPOT-1234")
        assert config.device_id == "SPOT-1234"
        assert config.transmission_mode == TransmissionMode.HYBRID

    def test_config_with_custom_values(self):
        """Test config with custom values."""
        config = SpotterConfig(
            device_id="SPOT-5678",
            transmission_mode=TransmissionMode.SATELLITE,
            sample_interval_seconds=120,
            max_queue_size=500,
        )
        assert config.transmission_mode == TransmissionMode.SATELLITE
        assert config.sample_interval_seconds == 120
        assert config.max_queue_size == 500


class TestSpotterBuoyClient:
    """Tests for SpotterBuoyClient class."""

    def test_create_client(self):
        """Test creating a client."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        assert client.device_id == "SPOT-1234"
        assert client.state == ConnectionState.DISCONNECTED

    def test_create_client_with_config(self):
        """Test creating client with custom config."""
        config = SpotterConfig(
            device_id="SPOT-1234",
            transmission_mode=TransmissionMode.CELLULAR,
        )
        client = SpotterBuoyClient(device_id="SPOT-1234", config=config)
        assert client.config.transmission_mode == TransmissionMode.CELLULAR

    def test_register_sensor(self):
        """Test registering a sensor."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        sensor = TemperatureSensor(sensor_id="temp_01")
        client.register_sensor(sensor)
        assert "temp_01" in client.list_sensors()

    def test_unregister_sensor(self):
        """Test unregistering a sensor."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        sensor = TemperatureSensor(sensor_id="temp_01")
        client.register_sensor(sensor)
        result = client.unregister_sensor("temp_01")
        assert result is True
        assert "temp_01" not in client.list_sensors()

    def test_unregister_nonexistent_sensor(self):
        """Test unregistering a sensor that doesn't exist."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        result = client.unregister_sensor("nonexistent")
        assert result is False

    def test_get_sensor(self):
        """Test getting a registered sensor."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        sensor = TemperatureSensor(sensor_id="temp_01")
        client.register_sensor(sensor)
        retrieved = client.get_sensor("temp_01")
        assert retrieved is sensor

    def test_get_nonexistent_sensor(self):
        """Test getting a sensor that doesn't exist."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        assert client.get_sensor("nonexistent") is None

    def test_list_sensors(self):
        """Test listing registered sensors."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        client.register_sensor(TemperatureSensor(sensor_id="temp_01"))
        client.register_sensor(PressureSensor(sensor_id="pressure_01"))
        sensors = client.list_sensors()
        assert len(sensors) == 2
        assert "temp_01" in sensors
        assert "pressure_01" in sensors


class TestSpotterBuoyClientQueue:
    """Tests for SpotterBuoyClient queue operations."""

    def test_queue_reading(self):
        """Test queuing a sensor reading."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        sensor = TemperatureSensor()
        reading = sensor.create_reading(22.5)
        msg = client.queue_reading(sensor, reading)
        assert client.get_queue_size() == 1
        assert isinstance(msg, BristlemouthMessage)

    def test_queue_message(self):
        """Test queuing a pre-formatted message."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        topic = BristlemouthTopic(name="test")
        msg = BristlemouthMessage(topic=topic, payload=b"test")
        client.queue_message(msg)
        assert client.get_queue_size() == 1

    def test_queue_raw_data(self):
        """Test queuing raw data."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        topic = BristlemouthTopic(name="test/data")
        msg = client.queue_raw_data(topic, b"raw data bytes")
        assert client.get_queue_size() == 1
        assert msg.payload == b"raw data bytes"

    def test_get_queue_size(self):
        """Test getting queue size."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        assert client.get_queue_size() == 0
        sensor = TemperatureSensor()
        for i in range(5):
            reading = sensor.create_reading(20.0 + i)
            client.queue_reading(sensor, reading)
        assert client.get_queue_size() == 5

    def test_clear_queue(self):
        """Test clearing the queue."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        sensor = TemperatureSensor()
        for i in range(3):
            reading = sensor.create_reading(20.0 + i)
            client.queue_reading(sensor, reading)
        cleared = client.clear_queue()
        assert cleared == 3
        assert client.get_queue_size() == 0

    def test_peek_queue(self):
        """Test peeking at queue."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        sensor = TemperatureSensor()
        for i in range(5):
            reading = sensor.create_reading(20.0 + i)
            client.queue_reading(sensor, reading)
        peeked = client.peek_queue(3)
        assert len(peeked) == 3
        assert client.get_queue_size() == 5  # Queue unchanged

    def test_queue_max_size(self):
        """Test queue respects max size."""
        config = SpotterConfig(device_id="SPOT-1234", max_queue_size=5)
        client = SpotterBuoyClient(device_id="SPOT-1234", config=config)
        sensor = TemperatureSensor()
        for i in range(10):
            reading = sensor.create_reading(20.0 + i)
            client.queue_reading(sensor, reading)
        assert client.get_queue_size() == 5  # Oldest messages dropped


class TestSpotterBuoyClientTransmission:
    """Tests for SpotterBuoyClient transmission."""

    def test_transmit_empty_queue(self):
        """Test transmitting with empty queue."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        result = client.transmit()
        assert result.success is True
        assert result.message_count == 0
        assert result.bytes_transmitted == 0

    def test_transmit_messages(self):
        """Test transmitting messages."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        sensor = TemperatureSensor()
        for i in range(3):
            reading = sensor.create_reading(20.0 + i)
            client.queue_reading(sensor, reading)
        result = client.transmit()
        assert result.success is True
        assert result.message_count == 3
        assert result.bytes_transmitted > 0
        assert client.get_queue_size() == 0

    def test_transmit_with_max_messages(self):
        """Test transmitting with max_messages limit."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        sensor = TemperatureSensor()
        for i in range(5):
            reading = sensor.create_reading(20.0 + i)
            client.queue_reading(sensor, reading)
        result = client.transmit(max_messages=2)
        assert result.message_count == 2
        assert client.get_queue_size() == 3

    def test_transmit_with_mode(self):
        """Test transmitting with specific mode."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        sensor = TemperatureSensor()
        reading = sensor.create_reading(22.5)
        client.queue_reading(sensor, reading)
        result = client.transmit(mode=TransmissionMode.SATELLITE)
        assert result.transmission_mode == TransmissionMode.SATELLITE

    def test_transmission_history(self):
        """Test transmission history is recorded."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        sensor = TemperatureSensor()
        reading = sensor.create_reading(22.5)
        client.queue_reading(sensor, reading)
        client.transmit()
        history = client.get_transmission_history()
        assert len(history) == 1
        assert history[0].success is True

    def test_transmission_history_limit(self):
        """Test transmission history with limit."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        sensor = TemperatureSensor()
        for i in range(5):
            reading = sensor.create_reading(20.0 + i)
            client.queue_reading(sensor, reading)
            client.transmit()
        history = client.get_transmission_history(limit=3)
        assert len(history) == 3


class TestSpotterBuoyClientCallbacks:
    """Tests for SpotterBuoyClient callbacks."""

    def test_on_transmit_callback(self):
        """Test on_transmit callback is called."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        callback_results = []

        def on_transmit(result):
            callback_results.append(result)

        client.on_transmit(on_transmit)
        sensor = TemperatureSensor()
        reading = sensor.create_reading(22.5)
        client.queue_reading(sensor, reading)
        client.transmit()
        assert len(callback_results) == 1
        assert callback_results[0].success is True

    def test_on_state_change_callback(self):
        """Test on_state_change callback is called."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        state_changes = []

        def on_state_change(old_state, new_state):
            state_changes.append((old_state, new_state))

        client.on_state_change(on_state_change)
        client.state = ConnectionState.CONNECTING
        client.state = ConnectionState.CONNECTED
        assert len(state_changes) == 2
        assert state_changes[0] == (ConnectionState.DISCONNECTED, ConnectionState.CONNECTING)
        assert state_changes[1] == (ConnectionState.CONNECTING, ConnectionState.CONNECTED)


class TestSpotterBuoyClientStatus:
    """Tests for SpotterBuoyClient status and formatting."""

    def test_create_status_message(self):
        """Test creating a status message."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        client.register_sensor(TemperatureSensor(sensor_id="temp_01"))
        msg = client.create_status_message()
        assert msg.message_type == MessageType.SENSOR_DATA
        status = json.loads(msg.payload.decode('utf-8'))
        assert status["device_id"] == "SPOT-1234"
        assert "temp_01" in status["registered_sensors"]

    def test_format_for_api(self):
        """Test formatting messages for API."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        sensor = TemperatureSensor()
        reading = sensor.create_reading(22.5)
        msg = sensor.wrap_reading(reading)
        api_data = client.format_for_api([msg])
        assert api_data["device_id"] == "SPOT-1234"
        assert api_data["message_count"] == 1
        assert len(api_data["messages"]) == 1
        assert "payload_hex" in api_data["messages"][0]

    def test_repr(self):
        """Test string representation."""
        client = SpotterBuoyClient(device_id="SPOT-1234")
        client.register_sensor(TemperatureSensor())
        repr_str = repr(client)
        assert "SPOT-1234" in repr_str
        assert "sensors=1" in repr_str


class TestCreateSpotterClient:
    """Tests for create_spotter_client factory function."""

    def test_create_with_defaults(self):
        """Test creating client with defaults."""
        client = create_spotter_client("SPOT-1234")
        assert client.device_id == "SPOT-1234"
        assert client.config.transmission_mode == TransmissionMode.HYBRID

    def test_create_with_custom_mode(self):
        """Test creating client with custom transmission mode."""
        client = create_spotter_client(
            "SPOT-1234",
            transmission_mode=TransmissionMode.CELLULAR,
        )
        assert client.config.transmission_mode == TransmissionMode.CELLULAR

    def test_create_with_custom_interval(self):
        """Test creating client with custom sample interval."""
        client = create_spotter_client(
            "SPOT-1234",
            sample_interval=120,
        )
        assert client.config.sample_interval_seconds == 120
