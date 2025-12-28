"""Tests for Bristlemouth protocol module."""

import pytest
import struct
import time

from ml_sensors.bristlemouth.protocol import (
    BristlemouthMessage,
    BristlemouthTopic,
    MessageType,
    DataType,
    SpotterTopics,
    BCMP_PROTOCOL_VERSION,
    BCMP_HEADER_MAGIC,
    MAX_PAYLOAD_SIZE,
    encode_sensor_value,
    decode_sensor_value,
)


class TestBristlemouthTopic:
    """Tests for BristlemouthTopic class."""

    def test_create_topic(self):
        """Test creating a topic."""
        topic = BristlemouthTopic(name="test/sensor")
        assert topic.name == "test/sensor"
        assert topic.version == 1

    def test_create_topic_with_version(self):
        """Test creating a topic with custom version."""
        topic = BristlemouthTopic(name="test/sensor", version=2)
        assert topic.version == 2

    def test_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            BristlemouthTopic(name="")

    def test_long_name_raises_error(self):
        """Test that name exceeding 64 chars raises ValueError."""
        with pytest.raises(ValueError, match="cannot exceed 64"):
            BristlemouthTopic(name="a" * 65)

    def test_to_bytes(self):
        """Test encoding topic to bytes."""
        topic = BristlemouthTopic(name="test")
        data = topic.to_bytes()
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_from_bytes(self):
        """Test decoding topic from bytes."""
        original = BristlemouthTopic(name="test/sensor", version=2)
        data = original.to_bytes()
        decoded = BristlemouthTopic.from_bytes(data)
        assert decoded.name == original.name
        assert decoded.version == original.version

    def test_str_representation(self):
        """Test string representation of topic."""
        topic = BristlemouthTopic(name="test/sensor", node_id="node1")
        assert "bm://" in str(topic)
        assert "test/sensor" in str(topic)


class TestSpotterTopics:
    """Tests for predefined Spotter topics."""

    def test_temperature_topic(self):
        """Test temperature topic exists."""
        assert SpotterTopics.TEMPERATURE.name == "spotter/sensor/temperature"

    def test_pressure_topic(self):
        """Test pressure topic exists."""
        assert SpotterTopics.PRESSURE.name == "spotter/sensor/pressure"

    def test_salinity_topic(self):
        """Test salinity topic exists."""
        assert SpotterTopics.SALINITY.name == "spotter/sensor/salinity"

    def test_transmit_data_topic(self):
        """Test transmit-data topic exists."""
        assert SpotterTopics.TRANSMIT_DATA.name == "spotter/transmit-data"


class TestBristlemouthMessage:
    """Tests for BristlemouthMessage class."""

    def test_create_message(self):
        """Test creating a message."""
        topic = BristlemouthTopic(name="test")
        msg = BristlemouthMessage(topic=topic, payload=b"hello")
        assert msg.topic == topic
        assert msg.payload == b"hello"
        assert msg.version == BCMP_PROTOCOL_VERSION

    def test_message_timestamp_auto_set(self):
        """Test that timestamp is automatically set."""
        topic = BristlemouthTopic(name="test")
        before = time.time()
        msg = BristlemouthMessage(topic=topic, payload=b"test")
        after = time.time()
        assert before <= msg.timestamp <= after

    def test_message_custom_timestamp(self):
        """Test message with custom timestamp."""
        topic = BristlemouthTopic(name="test")
        custom_time = 1234567890.0
        msg = BristlemouthMessage(topic=topic, payload=b"test", timestamp=custom_time)
        assert msg.timestamp == custom_time

    def test_payload_too_large_raises_error(self):
        """Test that oversized payload raises ValueError."""
        topic = BristlemouthTopic(name="test")
        with pytest.raises(ValueError, match="exceeds maximum"):
            BristlemouthMessage(topic=topic, payload=b"x" * (MAX_PAYLOAD_SIZE + 1))

    def test_to_bytes(self):
        """Test encoding message to bytes."""
        topic = BristlemouthTopic(name="test")
        msg = BristlemouthMessage(topic=topic, payload=b"hello")
        data = msg.to_bytes()
        assert isinstance(data, bytes)
        assert data[0] == BCMP_HEADER_MAGIC

    def test_from_bytes(self):
        """Test decoding message from bytes."""
        topic = BristlemouthTopic(name="test/sensor")
        original = BristlemouthMessage(
            topic=topic,
            payload=b"test data",
            message_type=MessageType.SENSOR_DATA,
            sequence_number=42,
        )
        data = original.to_bytes()
        decoded = BristlemouthMessage.from_bytes(data)
        assert decoded.topic.name == original.topic.name
        assert decoded.payload == original.payload
        assert decoded.message_type == original.message_type
        assert decoded.sequence_number == original.sequence_number

    def test_roundtrip_encoding(self):
        """Test roundtrip encoding/decoding."""
        topic = BristlemouthTopic(name="spotter/sensor/temperature")
        original = BristlemouthMessage(
            topic=topic,
            payload=struct.pack('!f', 22.5),
            message_type=MessageType.SENSOR_DATA,
        )
        data = original.to_bytes()
        decoded = BristlemouthMessage.from_bytes(data)
        assert struct.unpack('!f', decoded.payload)[0] == pytest.approx(22.5)

    def test_invalid_magic_byte_raises_error(self):
        """Test that invalid magic byte raises ValueError."""
        topic = BristlemouthTopic(name="test")
        msg = BristlemouthMessage(topic=topic, payload=b"test")
        data = bytearray(msg.to_bytes())
        data[0] = 0x00  # Invalid magic byte
        with pytest.raises(ValueError, match="Invalid magic byte"):
            BristlemouthMessage.from_bytes(bytes(data))

    def test_checksum_mismatch_raises_error(self):
        """Test that checksum mismatch raises ValueError."""
        topic = BristlemouthTopic(name="test")
        msg = BristlemouthMessage(topic=topic, payload=b"test")
        data = bytearray(msg.to_bytes())
        data[-1] ^= 0xFF  # Corrupt checksum
        with pytest.raises(ValueError, match="Checksum mismatch"):
            BristlemouthMessage.from_bytes(bytes(data))

    def test_to_hex(self):
        """Test converting message to hex string."""
        topic = BristlemouthTopic(name="test")
        msg = BristlemouthMessage(topic=topic, payload=b"hi")
        hex_str = msg.to_hex()
        assert isinstance(hex_str, str)
        assert all(c in '0123456789abcdef' for c in hex_str)

    def test_from_hex(self):
        """Test creating message from hex string."""
        topic = BristlemouthTopic(name="test")
        original = BristlemouthMessage(topic=topic, payload=b"hello")
        hex_str = original.to_hex()
        decoded = BristlemouthMessage.from_hex(hex_str)
        assert decoded.payload == original.payload

    def test_to_dict(self):
        """Test converting message to dictionary."""
        topic = BristlemouthTopic(name="test")
        msg = BristlemouthMessage(
            topic=topic,
            payload=b"test",
            message_type=MessageType.SENSOR_DATA,
        )
        d = msg.to_dict()
        assert "topic" in d
        assert "message_type" in d
        assert d["message_type"] == "SENSOR_DATA"
        assert "payload_hex" in d


class TestEncodeSensorValue:
    """Tests for encode_sensor_value function."""

    def test_encode_float32(self):
        """Test encoding float32."""
        data = encode_sensor_value(3.14, DataType.FLOAT32)
        assert len(data) == 4
        decoded = struct.unpack('!f', data)[0]
        assert decoded == pytest.approx(3.14, rel=1e-6)

    def test_encode_float64(self):
        """Test encoding float64."""
        data = encode_sensor_value(3.14159265359, DataType.FLOAT64)
        assert len(data) == 8
        decoded = struct.unpack('!d', data)[0]
        assert decoded == pytest.approx(3.14159265359)

    def test_encode_int32(self):
        """Test encoding int32."""
        data = encode_sensor_value(-12345, DataType.INT32)
        assert len(data) == 4
        decoded = struct.unpack('!i', data)[0]
        assert decoded == -12345

    def test_encode_uint16(self):
        """Test encoding uint16."""
        data = encode_sensor_value(65000, DataType.UINT16)
        assert len(data) == 2
        decoded = struct.unpack('!H', data)[0]
        assert decoded == 65000

    def test_encode_string(self):
        """Test encoding string."""
        data = encode_sensor_value("hello", DataType.STRING)
        length = struct.unpack('!H', data[:2])[0]
        assert length == 5
        assert data[2:].decode('utf-8') == "hello"

    def test_encode_bool_true(self):
        """Test encoding boolean true."""
        data = encode_sensor_value(True, DataType.BOOL)
        assert len(data) == 1
        assert struct.unpack('!?', data)[0] is True

    def test_encode_bool_false(self):
        """Test encoding boolean false."""
        data = encode_sensor_value(False, DataType.BOOL)
        assert struct.unpack('!?', data)[0] is False

    def test_encode_bytes(self):
        """Test encoding bytes."""
        original = b'\x01\x02\x03\x04'
        data = encode_sensor_value(original, DataType.BYTES)
        length = struct.unpack('!H', data[:2])[0]
        assert length == 4
        assert data[2:] == original


class TestDecodeSensorValue:
    """Tests for decode_sensor_value function."""

    def test_decode_float32(self):
        """Test decoding float32."""
        data = struct.pack('!f', 2.718)
        value, consumed = decode_sensor_value(data, DataType.FLOAT32)
        assert value == pytest.approx(2.718, rel=1e-6)
        assert consumed == 4

    def test_decode_float64(self):
        """Test decoding float64."""
        data = struct.pack('!d', 2.718281828)
        value, consumed = decode_sensor_value(data, DataType.FLOAT64)
        assert value == pytest.approx(2.718281828)
        assert consumed == 8

    def test_decode_int16(self):
        """Test decoding int16."""
        data = struct.pack('!h', -1000)
        value, consumed = decode_sensor_value(data, DataType.INT16)
        assert value == -1000
        assert consumed == 2

    def test_decode_string(self):
        """Test decoding string."""
        text = "world"
        data = struct.pack('!H', len(text)) + text.encode('utf-8')
        value, consumed = decode_sensor_value(data, DataType.STRING)
        assert value == "world"
        assert consumed == 2 + len(text)

    def test_roundtrip_all_types(self):
        """Test roundtrip encoding/decoding for all types."""
        test_cases = [
            (3.14, DataType.FLOAT32),
            (3.14159265359, DataType.FLOAT64),
            (-128, DataType.INT8),
            (-32000, DataType.INT16),
            (-2000000, DataType.INT32),
            (255, DataType.UINT8),
            (65000, DataType.UINT16),
            (True, DataType.BOOL),
            ("test string", DataType.STRING),
        ]
        for value, dtype in test_cases:
            encoded = encode_sensor_value(value, dtype)
            decoded, _ = decode_sensor_value(encoded, dtype)
            if dtype in (DataType.FLOAT32, DataType.FLOAT64):
                assert decoded == pytest.approx(value, rel=1e-6)
            else:
                assert decoded == value
