"""Tests for ML Sensor Datasheet class."""

import pytest
from ml_sensors.datasheet import MLSensorDatasheet


class TestMLSensorDatasheet:
    """Tests for MLSensorDatasheet class."""

    def test_create_datasheet_with_name(self):
        """Test creating a datasheet with just a name."""
        ds = MLSensorDatasheet(name="Person Detection Sensor")
        assert ds.name == "Person Detection Sensor"
        assert ds.version == "1.0.0"

    def test_create_datasheet_with_all_fields(self):
        """Test creating a datasheet with all fields."""
        ds = MLSensorDatasheet(
            name="Person Detection Sensor",
            version="2.0.0",
            description="Detects human presence",
            power_consumption_mw=150.0,
            latency_ms=50.0,
            accuracy_metrics={"accuracy": 0.95, "precision": 0.92, "recall": 0.88},
            min_temp_c=-20.0,
            max_temp_c=60.0,
            input_type="image",
            output_classes=["person", "no_person"],
        )
        assert ds.name == "Person Detection Sensor"
        assert ds.version == "2.0.0"
        assert ds.power_consumption_mw == 150.0
        assert ds.latency_ms == 50.0
        assert ds.accuracy_metrics["accuracy"] == 0.95
        assert ds.input_type == "image"
        assert len(ds.output_classes) == 2


class TestMLSensorDatasheetValidation:
    """Tests for MLSensorDatasheet validation."""

    def test_validate_valid_datasheet(self):
        """Test validation of a valid datasheet."""
        ds = MLSensorDatasheet(
            name="Test Sensor",
            power_consumption_mw=100.0,
            latency_ms=50.0,
            accuracy_metrics={"accuracy": 0.95, "precision": 0.92, "recall": 0.88},
            min_temp_c=-20.0,
            max_temp_c=60.0,
        )
        is_valid, errors = ds.validate()
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_empty_name(self):
        """Test validation fails with empty name."""
        ds = MLSensorDatasheet(name="")
        is_valid, errors = ds.validate()
        assert is_valid is False
        assert any("name" in e.lower() for e in errors)

    def test_validate_negative_power(self):
        """Test validation fails with negative power consumption."""
        ds = MLSensorDatasheet(name="Test", power_consumption_mw=-10.0)
        is_valid, errors = ds.validate()
        assert is_valid is False
        assert any("power" in e.lower() for e in errors)

    def test_validate_negative_latency(self):
        """Test validation fails with negative latency."""
        ds = MLSensorDatasheet(name="Test", latency_ms=-5.0)
        is_valid, errors = ds.validate()
        assert is_valid is False
        assert any("latency" in e.lower() for e in errors)

    def test_validate_invalid_accuracy_metrics(self):
        """Test validation fails with invalid accuracy metrics."""
        ds = MLSensorDatasheet(
            name="Test",
            accuracy_metrics={"accuracy": 1.5},  # Out of range
        )
        is_valid, errors = ds.validate()
        assert is_valid is False
        assert any("accuracy" in e.lower() for e in errors)

    def test_validate_invalid_temperature_range(self):
        """Test validation fails with invalid temperature range."""
        ds = MLSensorDatasheet(name="Test", min_temp_c=50.0, max_temp_c=25.0)
        is_valid, errors = ds.validate()
        assert is_valid is False
        assert any("temperature" in e.lower() for e in errors)

    def test_validate_multiple_errors(self):
        """Test validation collects multiple errors."""
        ds = MLSensorDatasheet(
            name="",
            power_consumption_mw=-10.0,
            latency_ms=-5.0,
        )
        is_valid, errors = ds.validate()
        assert is_valid is False
        assert len(errors) >= 3


class TestMLSensorDatasheetEfficiency:
    """Tests for MLSensorDatasheet efficiency score."""

    def test_efficiency_score_calculation(self):
        """Test efficiency score calculation."""
        ds = MLSensorDatasheet(
            name="Test",
            power_consumption_mw=100.0,  # 90% power score
            latency_ms=100.0,  # 90% latency score
        )
        score = ds.get_efficiency_score()
        assert 0.89 <= score <= 0.91  # Should be around 0.9

    def test_efficiency_score_zero_power(self):
        """Test efficiency score with zero power."""
        ds = MLSensorDatasheet(name="Test", power_consumption_mw=0.0, latency_ms=50.0)
        score = ds.get_efficiency_score()
        assert score == 0.0

    def test_efficiency_score_zero_latency(self):
        """Test efficiency score with zero latency."""
        ds = MLSensorDatasheet(name="Test", power_consumption_mw=50.0, latency_ms=0.0)
        score = ds.get_efficiency_score()
        assert score == 0.0

    def test_efficiency_score_high_values(self):
        """Test efficiency score with high power and latency."""
        ds = MLSensorDatasheet(
            name="Test",
            power_consumption_mw=900.0,
            latency_ms=900.0,
        )
        score = ds.get_efficiency_score()
        assert 0.09 <= score <= 0.11  # Should be around 0.1

    def test_efficiency_score_max_values(self):
        """Test efficiency score at maximum values."""
        ds = MLSensorDatasheet(
            name="Test",
            power_consumption_mw=1000.0,
            latency_ms=1000.0,
        )
        score = ds.get_efficiency_score()
        assert score == 0.0


class TestMLSensorDatasheetAccuracy:
    """Tests for MLSensorDatasheet accuracy methods."""

    def test_get_overall_accuracy(self):
        """Test getting overall accuracy."""
        ds = MLSensorDatasheet(
            name="Test",
            accuracy_metrics={"accuracy": 0.95, "precision": 0.92, "recall": 0.88},
        )
        accuracy = ds.get_overall_accuracy()
        assert accuracy == 0.95

    def test_get_overall_accuracy_missing(self):
        """Test getting overall accuracy when not present."""
        ds = MLSensorDatasheet(
            name="Test",
            accuracy_metrics={"precision": 0.92, "recall": 0.88},
        )
        accuracy = ds.get_overall_accuracy()
        assert accuracy is None

    def test_get_overall_accuracy_empty_metrics(self):
        """Test getting overall accuracy with empty metrics."""
        ds = MLSensorDatasheet(name="Test")
        accuracy = ds.get_overall_accuracy()
        assert accuracy is None


class TestMLSensorDatasheetSerialization:
    """Tests for MLSensorDatasheet serialization."""

    def test_to_dict(self):
        """Test converting datasheet to dictionary."""
        ds = MLSensorDatasheet(
            name="Test Sensor",
            version="1.0.0",
            description="Test description",
            power_consumption_mw=100.0,
            latency_ms=50.0,
            accuracy_metrics={"accuracy": 0.95},
            min_temp_c=-20.0,
            max_temp_c=60.0,
            input_type="image",
            output_classes=["a", "b"],
        )
        data = ds.to_dict()
        assert data["name"] == "Test Sensor"
        assert data["version"] == "1.0.0"
        assert data["power_consumption_mw"] == 100.0
        assert data["accuracy_metrics"]["accuracy"] == 0.95
        assert data["output_classes"] == ["a", "b"]

    def test_from_dict(self):
        """Test creating datasheet from dictionary."""
        data = {
            "name": "Test Sensor",
            "version": "2.0.0",
            "description": "From dict",
            "power_consumption_mw": 200.0,
            "latency_ms": 75.0,
            "accuracy_metrics": {"accuracy": 0.90},
            "min_temp_c": -10.0,
            "max_temp_c": 50.0,
            "input_type": "audio",
            "output_classes": ["yes", "no"],
        }
        ds = MLSensorDatasheet.from_dict(data)
        assert ds.name == "Test Sensor"
        assert ds.version == "2.0.0"
        assert ds.power_consumption_mw == 200.0
        assert ds.input_type == "audio"

    def test_from_dict_with_defaults(self):
        """Test creating datasheet from partial dictionary."""
        data = {"name": "Minimal Sensor"}
        ds = MLSensorDatasheet.from_dict(data)
        assert ds.name == "Minimal Sensor"
        assert ds.version == "1.0.0"
        assert ds.power_consumption_mw == 0.0

    def test_roundtrip_serialization(self):
        """Test roundtrip serialization."""
        original = MLSensorDatasheet(
            name="Roundtrip Test",
            version="3.0.0",
            power_consumption_mw=150.0,
            latency_ms=25.0,
            accuracy_metrics={"accuracy": 0.99, "precision": 0.98, "recall": 0.97},
        )
        data = original.to_dict()
        restored = MLSensorDatasheet.from_dict(data)
        assert restored.name == original.name
        assert restored.version == original.version
        assert restored.power_consumption_mw == original.power_consumption_mw
        assert restored.accuracy_metrics == original.accuracy_metrics
