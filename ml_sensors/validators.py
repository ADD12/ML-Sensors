"""Validation utilities for ML Sensor specifications."""

from typing import Dict, Tuple, Union


def validate_power_consumption(
    power_mw: float, max_power_mw: float = 1000.0
) -> Tuple[bool, str]:
    """
    Validate power consumption value for an ML sensor.

    Args:
        power_mw: Power consumption in milliwatts
        max_power_mw: Maximum acceptable power consumption (default: 1000mW)

    Returns:
        Tuple of (is_valid, message)
    """
    if not isinstance(power_mw, (int, float)):
        return False, "Power consumption must be a numeric value"

    if power_mw < 0:
        return False, "Power consumption cannot be negative"

    if power_mw > max_power_mw:
        return False, f"Power consumption {power_mw}mW exceeds maximum {max_power_mw}mW"

    return True, "Valid power consumption"


def validate_accuracy_metrics(
    metrics: Dict[str, float], required_keys: Tuple[str, ...] = ("accuracy", "precision", "recall")
) -> Tuple[bool, str]:
    """
    Validate accuracy metrics for an ML sensor model.

    Args:
        metrics: Dictionary containing accuracy metrics
        required_keys: Tuple of required metric keys

    Returns:
        Tuple of (is_valid, message)
    """
    if not isinstance(metrics, dict):
        return False, "Metrics must be a dictionary"

    missing_keys = [key for key in required_keys if key not in metrics]
    if missing_keys:
        return False, f"Missing required metrics: {', '.join(missing_keys)}"

    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            return False, f"Metric '{key}' must be a numeric value"

        if not 0.0 <= value <= 1.0:
            return False, f"Metric '{key}' must be between 0.0 and 1.0"

    return True, "Valid accuracy metrics"


def validate_latency(
    latency_ms: float, max_latency_ms: float = 1000.0
) -> Tuple[bool, str]:
    """
    Validate inference latency for an ML sensor.

    Args:
        latency_ms: Inference latency in milliseconds
        max_latency_ms: Maximum acceptable latency (default: 1000ms)

    Returns:
        Tuple of (is_valid, message)
    """
    if not isinstance(latency_ms, (int, float)):
        return False, "Latency must be a numeric value"

    if latency_ms < 0:
        return False, "Latency cannot be negative"

    if latency_ms > max_latency_ms:
        return False, f"Latency {latency_ms}ms exceeds maximum {max_latency_ms}ms"

    return True, "Valid latency"


def validate_temperature_range(
    min_temp_c: float, max_temp_c: float, operating_range: Tuple[float, float] = (-40.0, 85.0)
) -> Tuple[bool, str]:
    """
    Validate operating temperature range for an ML sensor.

    Args:
        min_temp_c: Minimum operating temperature in Celsius
        max_temp_c: Maximum operating temperature in Celsius
        operating_range: Acceptable operating range (default: -40°C to 85°C)

    Returns:
        Tuple of (is_valid, message)
    """
    if not isinstance(min_temp_c, (int, float)) or not isinstance(max_temp_c, (int, float)):
        return False, "Temperature values must be numeric"

    if min_temp_c >= max_temp_c:
        return False, "Minimum temperature must be less than maximum temperature"

    if min_temp_c < operating_range[0]:
        return False, f"Minimum temperature {min_temp_c}°C is below acceptable range {operating_range[0]}°C"

    if max_temp_c > operating_range[1]:
        return False, f"Maximum temperature {max_temp_c}°C exceeds acceptable range {operating_range[1]}°C"

    return True, "Valid temperature range"
