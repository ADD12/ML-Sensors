"""Tests for ML Sensor validators."""

import pytest
from ml_sensors.validators import (
    validate_power_consumption,
    validate_accuracy_metrics,
    validate_latency,
    validate_temperature_range,
)


class TestValidatePowerConsumption:
    """Tests for validate_power_consumption function."""

    def test_valid_power_consumption(self):
        """Test valid power consumption values."""
        is_valid, msg = validate_power_consumption(100.0)
        assert is_valid is True
        assert msg == "Valid power consumption"

    def test_zero_power_consumption(self):
        """Test zero power consumption is valid."""
        is_valid, msg = validate_power_consumption(0.0)
        assert is_valid is True

    def test_negative_power_consumption(self):
        """Test negative power consumption is invalid."""
        is_valid, msg = validate_power_consumption(-10.0)
        assert is_valid is False
        assert "cannot be negative" in msg

    def test_exceeds_max_power(self):
        """Test power consumption exceeding maximum."""
        is_valid, msg = validate_power_consumption(1500.0, max_power_mw=1000.0)
        assert is_valid is False
        assert "exceeds maximum" in msg

    def test_custom_max_power(self):
        """Test with custom maximum power."""
        is_valid, msg = validate_power_consumption(50.0, max_power_mw=100.0)
        assert is_valid is True

    def test_invalid_type(self):
        """Test non-numeric power consumption."""
        is_valid, msg = validate_power_consumption("100")
        assert is_valid is False
        assert "numeric value" in msg

    def test_integer_power_consumption(self):
        """Test integer power consumption is valid."""
        is_valid, msg = validate_power_consumption(100)
        assert is_valid is True


class TestValidateAccuracyMetrics:
    """Tests for validate_accuracy_metrics function."""

    def test_valid_metrics(self):
        """Test valid accuracy metrics."""
        metrics = {"accuracy": 0.95, "precision": 0.92, "recall": 0.88}
        is_valid, msg = validate_accuracy_metrics(metrics)
        assert is_valid is True
        assert msg == "Valid accuracy metrics"

    def test_missing_required_keys(self):
        """Test metrics missing required keys."""
        metrics = {"accuracy": 0.95}
        is_valid, msg = validate_accuracy_metrics(metrics)
        assert is_valid is False
        assert "Missing required metrics" in msg

    def test_custom_required_keys(self):
        """Test with custom required keys."""
        metrics = {"f1_score": 0.90}
        is_valid, msg = validate_accuracy_metrics(metrics, required_keys=("f1_score",))
        assert is_valid is True

    def test_metric_out_of_range_high(self):
        """Test metric value above 1.0."""
        metrics = {"accuracy": 1.5, "precision": 0.92, "recall": 0.88}
        is_valid, msg = validate_accuracy_metrics(metrics)
        assert is_valid is False
        assert "between 0.0 and 1.0" in msg

    def test_metric_out_of_range_low(self):
        """Test metric value below 0.0."""
        metrics = {"accuracy": -0.1, "precision": 0.92, "recall": 0.88}
        is_valid, msg = validate_accuracy_metrics(metrics)
        assert is_valid is False
        assert "between 0.0 and 1.0" in msg

    def test_non_numeric_metric(self):
        """Test non-numeric metric value."""
        metrics = {"accuracy": "high", "precision": 0.92, "recall": 0.88}
        is_valid, msg = validate_accuracy_metrics(metrics)
        assert is_valid is False
        assert "numeric value" in msg

    def test_invalid_metrics_type(self):
        """Test non-dictionary metrics."""
        is_valid, msg = validate_accuracy_metrics([0.95, 0.92, 0.88])
        assert is_valid is False
        assert "must be a dictionary" in msg

    def test_boundary_values(self):
        """Test boundary values 0.0 and 1.0."""
        metrics = {"accuracy": 0.0, "precision": 1.0, "recall": 0.5}
        is_valid, msg = validate_accuracy_metrics(metrics)
        assert is_valid is True


class TestValidateLatency:
    """Tests for validate_latency function."""

    def test_valid_latency(self):
        """Test valid latency value."""
        is_valid, msg = validate_latency(50.0)
        assert is_valid is True
        assert msg == "Valid latency"

    def test_zero_latency(self):
        """Test zero latency is valid."""
        is_valid, msg = validate_latency(0.0)
        assert is_valid is True

    def test_negative_latency(self):
        """Test negative latency is invalid."""
        is_valid, msg = validate_latency(-5.0)
        assert is_valid is False
        assert "cannot be negative" in msg

    def test_exceeds_max_latency(self):
        """Test latency exceeding maximum."""
        is_valid, msg = validate_latency(2000.0, max_latency_ms=1000.0)
        assert is_valid is False
        assert "exceeds maximum" in msg

    def test_custom_max_latency(self):
        """Test with custom maximum latency."""
        is_valid, msg = validate_latency(80.0, max_latency_ms=100.0)
        assert is_valid is True

    def test_invalid_type(self):
        """Test non-numeric latency."""
        is_valid, msg = validate_latency("fast")
        assert is_valid is False
        assert "numeric value" in msg


class TestValidateTemperatureRange:
    """Tests for validate_temperature_range function."""

    def test_valid_temperature_range(self):
        """Test valid temperature range."""
        is_valid, msg = validate_temperature_range(-20.0, 60.0)
        assert is_valid is True
        assert msg == "Valid temperature range"

    def test_min_equals_max(self):
        """Test min equals max is invalid."""
        is_valid, msg = validate_temperature_range(25.0, 25.0)
        assert is_valid is False
        assert "less than maximum" in msg

    def test_min_greater_than_max(self):
        """Test min greater than max is invalid."""
        is_valid, msg = validate_temperature_range(50.0, 25.0)
        assert is_valid is False
        assert "less than maximum" in msg

    def test_below_operating_range(self):
        """Test min below operating range."""
        is_valid, msg = validate_temperature_range(-50.0, 60.0)
        assert is_valid is False
        assert "below acceptable range" in msg

    def test_above_operating_range(self):
        """Test max above operating range."""
        is_valid, msg = validate_temperature_range(-20.0, 100.0)
        assert is_valid is False
        assert "exceeds acceptable range" in msg

    def test_custom_operating_range(self):
        """Test with custom operating range."""
        is_valid, msg = validate_temperature_range(0.0, 50.0, operating_range=(0.0, 50.0))
        assert is_valid is True

    def test_invalid_min_type(self):
        """Test non-numeric min temperature."""
        is_valid, msg = validate_temperature_range("cold", 60.0)
        assert is_valid is False
        assert "must be numeric" in msg

    def test_invalid_max_type(self):
        """Test non-numeric max temperature."""
        is_valid, msg = validate_temperature_range(-20.0, "hot")
        assert is_valid is False
        assert "must be numeric" in msg

    def test_boundary_values(self):
        """Test boundary values of default operating range."""
        is_valid, msg = validate_temperature_range(-40.0, 85.0)
        assert is_valid is True
