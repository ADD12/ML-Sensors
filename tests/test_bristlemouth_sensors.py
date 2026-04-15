"""Tests for Bristlemouth sensor classes."""

import pytest
import struct
import time

from ml_sensors.bristlemouth.sensors import (
    SensorReading,
    SensorUnit,
    TemperatureSensor,
    PressureSensor,
    SalinitySensor,
    CurrentMeterSensor,
    TurbiditySensor,
)
from ml_sensors.bristlemouth.protocol import MessageType


class TestSensorReading:
    """Tests for SensorReading class."""

    def test_create_reading(self):
        """Test creating a sensor reading."""
        reading = SensorReading(value=22.5, unit=SensorUnit.CELSIUS)
        assert reading.value == 22.5
        assert reading.unit == SensorUnit.CELSIUS
        assert reading.quality == 100

    def test_reading_with_all_fields(self):
        """Test creating a reading with all fields."""
        ts = time.time()
        reading = SensorReading(
            value=15.0,
            unit=SensorUnit.CELSIUS,
            timestamp=ts,
            quality=95,
            sensor_id="temp_01",
            depth_m=10.5,
        )
        assert reading.timestamp == ts
        assert reading.quality == 95
        assert reading.sensor_id == "temp_01"
        assert reading.depth_m == 10.5

    def test_invalid_quality_raises_error(self):
        """Test that invalid quality raises ValueError."""
        with pytest.raises(ValueError, match="between 0 and 100"):
            SensorReading(value=20.0, unit=SensorUnit.CELSIUS, quality=101)

    def test_negative_quality_raises_error(self):
        """Test that negative quality raises ValueError."""
        with pytest.raises(ValueError, match="between 0 and 100"):
            SensorReading(value=20.0, unit=SensorUnit.CELSIUS, quality=-1)

    def test_to_bytes(self):
        """Test encoding reading to bytes."""
        reading = SensorReading(value=22.5, unit=SensorUnit.CELSIUS, depth_m=5.0)
        data = reading.to_bytes()
        assert isinstance(data, bytes)
        assert len(data) == 21  # float64 + uint64 + uint8 + float32

    def test_from_bytes(self):
        """Test decoding reading from bytes."""
        original = SensorReading(
            value=22.5,
            unit=SensorUnit.CELSIUS,
            quality=90,
            depth_m=5.0,
        )
        data = original.to_bytes()
        decoded = SensorReading.from_bytes(data, SensorUnit.CELSIUS)
        assert decoded.value == pytest.approx(original.value)
        assert decoded.quality == original.quality
        assert decoded.depth_m == pytest.approx(original.depth_m)

    def test_to_dict(self):
        """Test converting reading to dictionary."""
        reading = SensorReading(
            value=22.5,
            unit=SensorUnit.CELSIUS,
            sensor_id="temp_01",
        )
        d = reading.to_dict()
        assert d["value"] == 22.5
        assert d["unit"] == "°C"
        assert d["sensor_id"] == "temp_01"


class TestTemperatureSensor:
    """Tests for TemperatureSensor class."""

    def test_create_sensor(self):
        """Test creating a temperature sensor."""
        sensor = TemperatureSensor()
        assert sensor.sensor_id == "temp_sensor_01"
        assert sensor.default_unit == SensorUnit.CELSIUS

    def test_create_sensor_with_custom_id(self):
        """Test creating sensor with custom ID."""
        sensor = TemperatureSensor(sensor_id="my_temp_sensor")
        assert sensor.sensor_id == "my_temp_sensor"

    def test_validate_valid_reading(self):
        """Test validating a valid temperature."""
        sensor = TemperatureSensor()
        is_valid, msg = sensor.validate_reading(22.5)
        assert is_valid is True

    def test_validate_below_minimum(self):
        """Test validating temperature below minimum."""
        sensor = TemperatureSensor(min_temp_c=-5.0)
        is_valid, msg = sensor.validate_reading(-10.0)
        assert is_valid is False
        assert "below minimum" in msg

    def test_validate_above_maximum(self):
        """Test validating temperature above maximum."""
        sensor = TemperatureSensor(max_temp_c=50.0)
        is_valid, msg = sensor.validate_reading(60.0)
        assert is_valid is False
        assert "above maximum" in msg

    def test_create_reading(self):
        """Test creating a reading through sensor."""
        sensor = TemperatureSensor()
        reading = sensor.create_reading(22.5)
        assert reading.value == 22.5
        assert reading.unit == SensorUnit.CELSIUS
        assert reading.sensor_id == sensor.sensor_id

    def test_create_reading_invalid_raises_error(self):
        """Test that invalid reading raises ValueError."""
        sensor = TemperatureSensor(max_temp_c=50.0)
        with pytest.raises(ValueError, match="Invalid reading"):
            sensor.create_reading(100.0)

    def test_wrap_reading(self):
        """Test wrapping reading in Bristlemouth message."""
        sensor = TemperatureSensor()
        reading = sensor.create_reading(22.5)
        msg = sensor.wrap_reading(reading)
        assert msg.message_type == MessageType.SENSOR_DATA
        assert "temperature" in msg.topic.name

    def test_create_and_wrap(self):
        """Test create_and_wrap convenience method."""
        sensor = TemperatureSensor()
        msg = sensor.create_and_wrap(22.5, depth_m=10.0)
        assert msg.message_type == MessageType.SENSOR_DATA

    def test_get_readings(self):
        """Test getting stored readings."""
        sensor = TemperatureSensor()
        sensor.create_reading(20.0)
        sensor.create_reading(21.0)
        sensor.create_reading(22.0)
        readings = sensor.get_readings()
        assert len(readings) == 3
        assert readings[0].value == 20.0

    def test_clear_readings(self):
        """Test clearing stored readings."""
        sensor = TemperatureSensor()
        sensor.create_reading(20.0)
        sensor.create_reading(21.0)
        sensor.clear_readings()
        assert len(sensor.get_readings()) == 0


class TestPressureSensor:
    """Tests for PressureSensor class."""

    def test_create_sensor(self):
        """Test creating a pressure sensor."""
        sensor = PressureSensor()
        assert sensor.default_unit == SensorUnit.DECIBAR

    def test_validate_valid_reading(self):
        """Test validating a valid pressure."""
        sensor = PressureSensor()
        is_valid, msg = sensor.validate_reading(100.0)
        assert is_valid is True

    def test_validate_negative_pressure(self):
        """Test validating negative pressure."""
        sensor = PressureSensor()
        is_valid, msg = sensor.validate_reading(-10.0)
        assert is_valid is False

    def test_pressure_to_depth(self):
        """Test pressure to depth conversion."""
        sensor = PressureSensor()
        depth = sensor.pressure_to_depth(100.0)  # 100 dbar
        assert 90 < depth < 110  # Approximately 100m

    def test_pressure_to_depth_at_different_latitudes(self):
        """Test depth varies with latitude."""
        sensor = PressureSensor()
        depth_equator = sensor.pressure_to_depth(100.0, latitude=0.0)
        depth_pole = sensor.pressure_to_depth(100.0, latitude=90.0)
        # Depth should be slightly different due to gravity variation
        assert depth_equator != depth_pole


class TestSalinitySensor:
    """Tests for SalinitySensor class."""

    def test_create_sensor(self):
        """Test creating a salinity sensor."""
        sensor = SalinitySensor()
        assert sensor.default_unit == SensorUnit.PSU

    def test_validate_valid_reading(self):
        """Test validating a valid salinity."""
        sensor = SalinitySensor()
        is_valid, msg = sensor.validate_reading(35.0)  # Typical ocean salinity
        assert is_valid is True

    def test_validate_negative_salinity(self):
        """Test validating negative salinity."""
        sensor = SalinitySensor()
        is_valid, msg = sensor.validate_reading(-1.0)
        assert is_valid is False

    def test_validate_above_maximum(self):
        """Test validating salinity above maximum."""
        sensor = SalinitySensor(max_salinity_psu=45.0)
        is_valid, msg = sensor.validate_reading(50.0)
        assert is_valid is False


class TestCurrentMeterSensor:
    """Tests for CurrentMeterSensor class."""

    def test_create_sensor(self):
        """Test creating a current meter sensor."""
        sensor = CurrentMeterSensor()
        assert sensor.default_unit == SensorUnit.METERS_PER_SECOND

    def test_validate_valid_reading(self):
        """Test validating a valid current speed."""
        sensor = CurrentMeterSensor()
        is_valid, msg = sensor.validate_reading(1.5)
        assert is_valid is True

    def test_validate_negative_speed(self):
        """Test validating negative speed."""
        sensor = CurrentMeterSensor()
        is_valid, msg = sensor.validate_reading(-0.5)
        assert is_valid is False

    def test_create_current_reading(self):
        """Test creating a current reading with direction."""
        sensor = CurrentMeterSensor()
        reading, direction = sensor.create_current_reading(1.5, 180.0)
        assert reading.value == 1.5
        assert direction == 180.0

    def test_invalid_direction_raises_error(self):
        """Test that invalid direction raises ValueError."""
        sensor = CurrentMeterSensor()
        with pytest.raises(ValueError, match="between 0 and 360"):
            sensor.create_current_reading(1.0, 400.0)

    def test_wrap_current_reading(self):
        """Test wrapping current reading with direction."""
        sensor = CurrentMeterSensor()
        reading, direction = sensor.create_current_reading(1.5, 90.0)
        msg = sensor.wrap_current_reading(reading, direction)
        # Payload should include direction
        assert len(msg.payload) > 21  # Reading bytes + direction


class TestTurbiditySensor:
    """Tests for TurbiditySensor class."""

    def test_create_sensor(self):
        """Test creating a turbidity sensor."""
        sensor = TurbiditySensor()
        assert sensor.default_unit == SensorUnit.NTU

    def test_validate_valid_reading(self):
        """Test validating a valid turbidity."""
        sensor = TurbiditySensor()
        is_valid, msg = sensor.validate_reading(10.0)
        assert is_valid is True

    def test_validate_negative_turbidity(self):
        """Test validating negative turbidity."""
        sensor = TurbiditySensor()
        is_valid, msg = sensor.validate_reading(-1.0)
        assert is_valid is False

    def test_classify_turbidity_excellent(self):
        """Test turbidity classification - excellent."""
        sensor = TurbiditySensor()
        assert sensor.classify_turbidity(0.5) == "Excellent"

    def test_classify_turbidity_good(self):
        """Test turbidity classification - good."""
        sensor = TurbiditySensor()
        assert sensor.classify_turbidity(3.0) == "Good"

    def test_classify_turbidity_fair(self):
        """Test turbidity classification - fair."""
        sensor = TurbiditySensor()
        assert sensor.classify_turbidity(25.0) == "Fair"

    def test_classify_turbidity_poor(self):
        """Test turbidity classification - poor."""
        sensor = TurbiditySensor()
        assert sensor.classify_turbidity(200.0) == "Poor"

    def test_classify_turbidity_very_poor(self):
        """Test turbidity classification - very poor."""
        sensor = TurbiditySensor()
        assert sensor.classify_turbidity(1000.0) == "Very Poor"
