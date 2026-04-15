"""ML Sensor Datasheet class for managing sensor specifications."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .validators import (
    validate_power_consumption,
    validate_accuracy_metrics,
    validate_latency,
    validate_temperature_range,
)


@dataclass
class MLSensorDatasheet:
    """
    Represents an ML Sensor datasheet with hardware and ML specifications.

    Attributes:
        name: Name of the ML sensor
        version: Version of the sensor/datasheet
        description: Brief description of the sensor's purpose
        power_consumption_mw: Power consumption in milliwatts
        latency_ms: Inference latency in milliseconds
        accuracy_metrics: Dictionary of accuracy metrics (accuracy, precision, recall, etc.)
        min_temp_c: Minimum operating temperature in Celsius
        max_temp_c: Maximum operating temperature in Celsius
        input_type: Type of input data (e.g., "image", "audio", "accelerometer")
        output_classes: List of output classification labels
    """

    name: str
    version: str = "1.0.0"
    description: str = ""
    power_consumption_mw: float = 0.0
    latency_ms: float = 0.0
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    min_temp_c: float = -40.0
    max_temp_c: float = 85.0
    input_type: str = "unknown"
    output_classes: List[str] = field(default_factory=list)

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate all datasheet specifications.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Validate name
        if not self.name or not isinstance(self.name, str):
            errors.append("Sensor name is required and must be a string")

        # Validate power consumption
        is_valid, msg = validate_power_consumption(self.power_consumption_mw)
        if not is_valid:
            errors.append(f"Power consumption: {msg}")

        # Validate latency
        is_valid, msg = validate_latency(self.latency_ms)
        if not is_valid:
            errors.append(f"Latency: {msg}")

        # Validate accuracy metrics if provided
        if self.accuracy_metrics:
            is_valid, msg = validate_accuracy_metrics(self.accuracy_metrics)
            if not is_valid:
                errors.append(f"Accuracy metrics: {msg}")

        # Validate temperature range
        is_valid, msg = validate_temperature_range(self.min_temp_c, self.max_temp_c)
        if not is_valid:
            errors.append(f"Temperature range: {msg}")

        return len(errors) == 0, errors

    def get_efficiency_score(self) -> float:
        """
        Calculate an efficiency score based on power and latency.

        Lower power and latency result in higher efficiency scores.
        Score is normalized between 0 and 1.

        Returns:
            Efficiency score between 0 and 1
        """
        if self.power_consumption_mw <= 0 or self.latency_ms <= 0:
            return 0.0

        # Normalize power (assuming max 1000mW)
        power_score = max(0, 1 - (self.power_consumption_mw / 1000.0))

        # Normalize latency (assuming max 1000ms)
        latency_score = max(0, 1 - (self.latency_ms / 1000.0))

        # Combined efficiency score
        return (power_score + latency_score) / 2

    def get_overall_accuracy(self) -> Optional[float]:
        """
        Get the overall accuracy from metrics if available.

        Returns:
            Accuracy value or None if not available
        """
        return self.accuracy_metrics.get("accuracy")

    def to_dict(self) -> Dict:
        """
        Convert datasheet to dictionary representation.

        Returns:
            Dictionary containing all datasheet fields
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "power_consumption_mw": self.power_consumption_mw,
            "latency_ms": self.latency_ms,
            "accuracy_metrics": self.accuracy_metrics,
            "min_temp_c": self.min_temp_c,
            "max_temp_c": self.max_temp_c,
            "input_type": self.input_type,
            "output_classes": self.output_classes,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MLSensorDatasheet":
        """
        Create a datasheet from a dictionary.

        Args:
            data: Dictionary containing datasheet fields

        Returns:
            MLSensorDatasheet instance
        """
        return cls(
            name=data.get("name", ""),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            power_consumption_mw=data.get("power_consumption_mw", 0.0),
            latency_ms=data.get("latency_ms", 0.0),
            accuracy_metrics=data.get("accuracy_metrics", {}),
            min_temp_c=data.get("min_temp_c", -40.0),
            max_temp_c=data.get("max_temp_c", 85.0),
            input_type=data.get("input_type", "unknown"),
            output_classes=data.get("output_classes", []),
        )
