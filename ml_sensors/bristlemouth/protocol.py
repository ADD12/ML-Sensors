"""
Bristlemouth Control Message Protocol (BCMP) implementation.

This module implements the core Bristlemouth protocol structures for
encoding and decoding sensor data messages.
"""

import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Union
import hashlib


# Bristlemouth protocol constants
BCMP_PROTOCOL_VERSION = 0x01
BCMP_HEADER_MAGIC = 0xBC
MAX_PAYLOAD_SIZE = 1400  # Maximum payload size in bytes (under typical MTU)


class MessageType(IntEnum):
    """Bristlemouth message types."""
    HEARTBEAT = 0x00
    SENSOR_DATA = 0x01
    CONFIG = 0x02
    ACK = 0x03
    NACK = 0x04
    DISCOVERY = 0x05
    SUBSCRIBE = 0x06
    UNSUBSCRIBE = 0x07
    PUBLISH = 0x08
    REQUEST = 0x09
    RESPONSE = 0x0A


class DataType(IntEnum):
    """Data type identifiers for sensor readings."""
    FLOAT32 = 0x01
    FLOAT64 = 0x02
    INT8 = 0x03
    INT16 = 0x04
    INT32 = 0x05
    INT64 = 0x06
    UINT8 = 0x07
    UINT16 = 0x08
    UINT32 = 0x09
    UINT64 = 0x0A
    STRING = 0x0B
    BYTES = 0x0C
    BOOL = 0x0D


@dataclass
class BristlemouthTopic:
    """
    Represents a Bristlemouth pub/sub topic.

    Topics are used for routing messages between nodes on the Bristlemouth network.
    Common topics include sensor data streams, configuration, and control messages.
    """
    name: str
    version: int = 1
    node_id: Optional[str] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("Topic name cannot be empty")
        if len(self.name) > 64:
            raise ValueError("Topic name cannot exceed 64 characters")

    def to_bytes(self) -> bytes:
        """Encode topic to bytes for transmission."""
        name_bytes = self.name.encode('utf-8')
        return struct.pack(
            f'!BH{len(name_bytes)}s',
            self.version,
            len(name_bytes),
            name_bytes
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'BristlemouthTopic':
        """Decode topic from bytes."""
        version = data[0]
        name_len = struct.unpack('!H', data[1:3])[0]
        name = data[3:3 + name_len].decode('utf-8')
        return cls(name=name, version=version)

    def __str__(self) -> str:
        return f"bm://{self.node_id or '*'}/{self.name}/v{self.version}"


# Common predefined topics for Spotter buoy
class SpotterTopics:
    """Predefined topics for Sofar Ocean Spotter buoy."""
    TEMPERATURE = BristlemouthTopic("spotter/sensor/temperature")
    PRESSURE = BristlemouthTopic("spotter/sensor/pressure")
    SALINITY = BristlemouthTopic("spotter/sensor/salinity")
    CURRENT = BristlemouthTopic("spotter/sensor/current")
    TURBIDITY = BristlemouthTopic("spotter/sensor/turbidity")
    WAVES = BristlemouthTopic("spotter/sensor/waves")
    GPS = BristlemouthTopic("spotter/sensor/gps")
    TRANSMIT_DATA = BristlemouthTopic("spotter/transmit-data")
    CONFIG = BristlemouthTopic("spotter/config")
    STATUS = BristlemouthTopic("spotter/status")


@dataclass
class BristlemouthMessage:
    """
    Bristlemouth protocol message.

    This class represents a complete Bristlemouth message with header,
    topic, payload, and checksum for reliable transmission.
    """
    topic: BristlemouthTopic
    payload: bytes
    message_type: MessageType = MessageType.SENSOR_DATA
    version: int = BCMP_PROTOCOL_VERSION
    sequence_number: int = 0
    timestamp: Optional[float] = None
    source_node_id: Optional[bytes] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if len(self.payload) > MAX_PAYLOAD_SIZE:
            raise ValueError(f"Payload size {len(self.payload)} exceeds maximum {MAX_PAYLOAD_SIZE}")

    def calculate_checksum(self) -> bytes:
        """Calculate CRC32 checksum of the message."""
        data = self._pack_without_checksum()
        crc = hashlib.md5(data).digest()[:4]
        return crc

    def _pack_without_checksum(self) -> bytes:
        """Pack message without checksum for checksum calculation."""
        topic_bytes = self.topic.to_bytes()
        source_id = self.source_node_id or b'\x00' * 6

        header = struct.pack(
            '!BBHIQ6s',
            BCMP_HEADER_MAGIC,
            self.version,
            self.message_type,
            self.sequence_number,
            int(self.timestamp * 1000),  # milliseconds
            source_id
        )

        return header + struct.pack('!H', len(topic_bytes)) + topic_bytes + \
               struct.pack('!H', len(self.payload)) + self.payload

    def to_bytes(self) -> bytes:
        """
        Encode message to bytes for transmission.

        Returns:
            Encoded message bytes including header, topic, payload, and checksum.
        """
        data = self._pack_without_checksum()
        checksum = self.calculate_checksum()
        return data + checksum

    @classmethod
    def from_bytes(cls, data: bytes) -> 'BristlemouthMessage':
        """
        Decode message from bytes.

        Args:
            data: Raw message bytes

        Returns:
            Decoded BristlemouthMessage

        Raises:
            ValueError: If message format is invalid or checksum fails
        """
        # Header size: magic(1) + version(1) + msg_type(2) + seq_num(4) + timestamp(8) + source_id(6) = 22
        header_size = 22
        if len(data) < header_size:
            raise ValueError("Message too short")

        magic, version, msg_type, seq_num, timestamp_ms, source_id = struct.unpack(
            '!BBHIQ6s', data[:header_size]
        )

        if magic != BCMP_HEADER_MAGIC:
            raise ValueError(f"Invalid magic byte: {magic:#x}")

        offset = header_size
        topic_len = struct.unpack('!H', data[offset:offset + 2])[0]
        offset += 2
        topic = BristlemouthTopic.from_bytes(data[offset:offset + topic_len])
        offset += topic_len

        payload_len = struct.unpack('!H', data[offset:offset + 2])[0]
        offset += 2
        payload = data[offset:offset + payload_len]
        offset += payload_len

        received_checksum = data[offset:offset + 4]

        msg = cls(
            topic=topic,
            payload=payload,
            message_type=MessageType(msg_type),
            version=version,
            sequence_number=seq_num,
            timestamp=timestamp_ms / 1000.0,
            source_node_id=source_id
        )

        expected_checksum = msg.calculate_checksum()
        if received_checksum != expected_checksum:
            raise ValueError("Checksum mismatch")

        return msg

    def to_hex(self) -> str:
        """Return message as hexadecimal string."""
        return self.to_bytes().hex()

    @classmethod
    def from_hex(cls, hex_string: str) -> 'BristlemouthMessage':
        """Create message from hexadecimal string."""
        return cls.from_bytes(bytes.fromhex(hex_string))

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            "topic": str(self.topic),
            "message_type": self.message_type.name,
            "version": self.version,
            "sequence_number": self.sequence_number,
            "timestamp": self.timestamp,
            "payload_hex": self.payload.hex(),
            "payload_size": len(self.payload),
        }


def encode_sensor_value(value: Union[int, float, str, bytes, bool], data_type: DataType) -> bytes:
    """
    Encode a sensor value to bytes based on data type.

    Args:
        value: The value to encode
        data_type: The data type for encoding

    Returns:
        Encoded bytes
    """
    if data_type == DataType.FLOAT32:
        return struct.pack('!f', float(value))
    elif data_type == DataType.FLOAT64:
        return struct.pack('!d', float(value))
    elif data_type == DataType.INT8:
        return struct.pack('!b', int(value))
    elif data_type == DataType.INT16:
        return struct.pack('!h', int(value))
    elif data_type == DataType.INT32:
        return struct.pack('!i', int(value))
    elif data_type == DataType.INT64:
        return struct.pack('!q', int(value))
    elif data_type == DataType.UINT8:
        return struct.pack('!B', int(value))
    elif data_type == DataType.UINT16:
        return struct.pack('!H', int(value))
    elif data_type == DataType.UINT32:
        return struct.pack('!I', int(value))
    elif data_type == DataType.UINT64:
        return struct.pack('!Q', int(value))
    elif data_type == DataType.STRING:
        encoded = str(value).encode('utf-8')
        return struct.pack('!H', len(encoded)) + encoded
    elif data_type == DataType.BYTES:
        if isinstance(value, bytes):
            return struct.pack('!H', len(value)) + value
        raise ValueError("Value must be bytes for BYTES data type")
    elif data_type == DataType.BOOL:
        return struct.pack('!?', bool(value))
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def decode_sensor_value(data: bytes, data_type: DataType) -> tuple:
    """
    Decode a sensor value from bytes.

    Args:
        data: The bytes to decode
        data_type: The data type for decoding

    Returns:
        Tuple of (decoded_value, bytes_consumed)
    """
    if data_type == DataType.FLOAT32:
        return struct.unpack('!f', data[:4])[0], 4
    elif data_type == DataType.FLOAT64:
        return struct.unpack('!d', data[:8])[0], 8
    elif data_type == DataType.INT8:
        return struct.unpack('!b', data[:1])[0], 1
    elif data_type == DataType.INT16:
        return struct.unpack('!h', data[:2])[0], 2
    elif data_type == DataType.INT32:
        return struct.unpack('!i', data[:4])[0], 4
    elif data_type == DataType.INT64:
        return struct.unpack('!q', data[:8])[0], 8
    elif data_type == DataType.UINT8:
        return struct.unpack('!B', data[:1])[0], 1
    elif data_type == DataType.UINT16:
        return struct.unpack('!H', data[:2])[0], 2
    elif data_type == DataType.UINT32:
        return struct.unpack('!I', data[:4])[0], 4
    elif data_type == DataType.UINT64:
        return struct.unpack('!Q', data[:8])[0], 8
    elif data_type == DataType.STRING:
        length = struct.unpack('!H', data[:2])[0]
        return data[2:2 + length].decode('utf-8'), 2 + length
    elif data_type == DataType.BYTES:
        length = struct.unpack('!H', data[:2])[0]
        return data[2:2 + length], 2 + length
    elif data_type == DataType.BOOL:
        return struct.unpack('!?', data[:1])[0], 1
    else:
        raise ValueError(f"Unknown data type: {data_type}")
