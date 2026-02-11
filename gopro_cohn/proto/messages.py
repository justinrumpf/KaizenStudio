# GoPro Protobuf Messages - Manual Implementation
# This avoids the need for protobuf compilation

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import struct


def _encode_varint(value: int) -> bytes:
    """Encode an integer as a protobuf varint."""
    bits = value & 0x7f
    value >>= 7
    result = b""
    while value:
        result += bytes([0x80 | bits])
        bits = value & 0x7f
        value >>= 7
    result += bytes([bits])
    return result


def _encode_field(field_num: int, wire_type: int, value: bytes) -> bytes:
    """Encode a protobuf field."""
    tag = (field_num << 3) | wire_type
    return _encode_varint(tag) + value


def _encode_string(field_num: int, value: str) -> bytes:
    """Encode a string field."""
    encoded = value.encode("utf-8")
    return _encode_field(field_num, 2, _encode_varint(len(encoded)) + encoded)


def _encode_bytes_field(field_num: int, value: bytes) -> bytes:
    """Encode a bytes field."""
    return _encode_field(field_num, 2, _encode_varint(len(value)) + value)


def _encode_bool(field_num: int, value: bool) -> bytes:
    """Encode a bool field."""
    return _encode_field(field_num, 0, _encode_varint(1 if value else 0))


def _encode_int32(field_num: int, value: int) -> bytes:
    """Encode an int32 field."""
    return _encode_field(field_num, 0, _encode_varint(value))


def _decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Decode a varint, return (value, new_position)."""
    result = 0
    shift = 0
    while True:
        byte = data[pos]
        result |= (byte & 0x7f) << shift
        pos += 1
        if not (byte & 0x80):
            break
        shift += 7
    return result, pos


def _decode_field(data: bytes, pos: int) -> tuple[int, int, bytes, int]:
    """Decode a field, return (field_num, wire_type, value_bytes, new_position)."""
    tag, pos = _decode_varint(data, pos)
    field_num = tag >> 3
    wire_type = tag & 0x7

    if wire_type == 0:  # Varint
        value, pos = _decode_varint(data, pos)
        return field_num, wire_type, _encode_varint(value), pos
    elif wire_type == 2:  # Length-delimited
        length, pos = _decode_varint(data, pos)
        value = data[pos : pos + length]
        return field_num, wire_type, value, pos + length
    else:
        raise ValueError(f"Unsupported wire type: {wire_type}")


# Enums
class EnumResultGeneric:
    RESULT_UNKNOWN = 0
    RESULT_SUCCESS = 1
    RESULT_ILL_FORMED = 2
    RESULT_NOT_SUPPORTED = 3
    RESULT_ARGUMENT_OUT_OF_BOUNDS = 4
    RESULT_ARGUMENT_INVALID = 5
    RESULT_RESOURCE_NOT_AVAILABLE = 6


class EnumCOHNStatus:
    COHN_UNPROVISIONED = 0
    COHN_PROVISIONED = 1


class EnumCOHNNetworkState:
    COHN_STATE_Init = 0
    COHN_STATE_Error = 1
    COHN_STATE_Exit = 2
    COHN_STATE_Idle = 5
    COHN_STATE_NetworkConnected = 27
    COHN_STATE_NetworkDisconnected = 28
    COHN_STATE_ConnectingToNetwork = 29
    COHN_STATE_Invalid = 30


class EnumProvisioning:
    PROVISIONING_UNKNOWN = 0
    PROVISIONING_NEVER_STARTED = 1
    PROVISIONING_STARTED = 2
    PROVISIONING_ABORTED_BY_SYSTEM = 3
    PROVISIONING_CANCELLED_BY_USER = 4
    PROVISIONING_SUCCESS_NEW_AP = 5
    PROVISIONING_SUCCESS_OLD_AP = 6
    PROVISIONING_ERROR_FAILED_TO_ASSOCIATE = 7
    PROVISIONING_ERROR_PASSWORD_AUTH = 8
    PROVISIONING_ERROR_EULA_BLOCKING = 9
    PROVISIONING_ERROR_NO_INTERNET = 10
    PROVISIONING_ERROR_UNSUPPORTED_TYPE = 11


class EnumScanning:
    SCANNING_UNKNOWN = 0
    SCANNING_NEVER_STARTED = 1
    SCANNING_STARTED = 2
    SCANNING_ABORTED_BY_SYSTEM = 3
    SCANNING_CANCELLED_BY_USER = 4
    SCANNING_SUCCESS = 5


class EnumScanEntryFlags:
    SCAN_FLAG_OPEN = 0x00
    SCAN_FLAG_AUTHENTICATED = 0x01
    SCAN_FLAG_CONFIGURED = 0x02
    SCAN_FLAG_BEST_SSID = 0x04
    SCAN_FLAG_ASSOCIATED = 0x08
    SCAN_FLAG_UNSUPPORTED_TYPE = 0x10


# Message classes
@dataclass
class ResponseGeneric:
    result: int = 0

    @classmethod
    def FromString(cls, data: bytes) -> "ResponseGeneric":
        obj = cls()
        pos = 0
        while pos < len(data):
            field_num, wire_type, value, pos = _decode_field(data, pos)
            if field_num == 1:
                obj.result, _ = _decode_varint(value, 0)
        return obj


@dataclass
class RequestStartScan:
    def SerializePartialToString(self) -> bytes:
        return b""


@dataclass
class RequestGetApEntries:
    start_index: int = 0
    max_entries: int = 0
    scan_id: int = 0

    def SerializePartialToString(self) -> bytes:
        result = b""
        result += _encode_int32(1, self.start_index)
        result += _encode_int32(2, self.max_entries)
        result += _encode_int32(3, self.scan_id)
        return result


@dataclass
class RequestConnect:
    ssid: str = ""

    def SerializePartialToString(self) -> bytes:
        return _encode_string(1, self.ssid)


@dataclass
class RequestConnectNew:
    ssid: str = ""
    password: str = ""

    def SerializePartialToString(self) -> bytes:
        result = b""
        result += _encode_string(1, self.ssid)
        result += _encode_string(2, self.password)
        return result


@dataclass
class ScanEntry:
    ssid: str = ""
    signal_strength_bars: int = 0
    signal_frequency_mhz: int = 0
    scan_entry_flags: int = 0

    @classmethod
    def FromBytes(cls, data: bytes) -> "ScanEntry":
        obj = cls()
        pos = 0
        while pos < len(data):
            field_num, wire_type, value, pos = _decode_field(data, pos)
            if field_num == 1:
                obj.ssid = value.decode("utf-8")
            elif field_num == 2:
                obj.signal_strength_bars, _ = _decode_varint(value, 0)
            elif field_num == 4:
                obj.signal_frequency_mhz, _ = _decode_varint(value, 0)
            elif field_num == 5:
                obj.scan_entry_flags, _ = _decode_varint(value, 0)
        return obj


@dataclass
class ResponseGetApEntries:
    result: int = 0
    scan_id: int = 0
    entries: List[ScanEntry] = field(default_factory=list)

    @classmethod
    def FromString(cls, data: bytes) -> "ResponseGetApEntries":
        obj = cls()
        pos = 0
        while pos < len(data):
            field_num, wire_type, value, pos = _decode_field(data, pos)
            if field_num == 1:
                obj.result, _ = _decode_varint(value, 0)
            elif field_num == 2:
                obj.scan_id, _ = _decode_varint(value, 0)
            elif field_num == 3:
                obj.entries.append(ScanEntry.FromBytes(value))
        return obj


@dataclass
class ResponseConnect:
    result: int = 0
    provisioning_state: int = 0
    timeout_seconds: int = 0

    @classmethod
    def FromString(cls, data: bytes) -> "ResponseConnect":
        obj = cls()
        pos = 0
        while pos < len(data):
            field_num, wire_type, value, pos = _decode_field(data, pos)
            if field_num == 1:
                obj.result, _ = _decode_varint(value, 0)
            elif field_num == 2:
                obj.provisioning_state, _ = _decode_varint(value, 0)
            elif field_num == 3:
                obj.timeout_seconds, _ = _decode_varint(value, 0)
        return obj


@dataclass
class ResponseConnectNew:
    result: int = 0
    provisioning_state: int = 0
    timeout_seconds: int = 0

    @classmethod
    def FromString(cls, data: bytes) -> "ResponseConnectNew":
        obj = cls()
        pos = 0
        while pos < len(data):
            field_num, wire_type, value, pos = _decode_field(data, pos)
            if field_num == 1:
                obj.result, _ = _decode_varint(value, 0)
            elif field_num == 2:
                obj.provisioning_state, _ = _decode_varint(value, 0)
            elif field_num == 3:
                obj.timeout_seconds, _ = _decode_varint(value, 0)
        return obj


@dataclass
class ResponseStartScanning:
    result: int = 0
    scanning_state: int = 0

    @classmethod
    def FromString(cls, data: bytes) -> "ResponseStartScanning":
        obj = cls()
        pos = 0
        while pos < len(data):
            field_num, wire_type, value, pos = _decode_field(data, pos)
            if field_num == 1:
                obj.result, _ = _decode_varint(value, 0)
            elif field_num == 2:
                obj.scanning_state, _ = _decode_varint(value, 0)
        return obj


@dataclass
class NotifStartScanning:
    scanning_state: int = 0
    scan_id: int = 0
    total_entries: int = 0
    total_configured_ssid: int = 0

    @classmethod
    def FromString(cls, data: bytes) -> "NotifStartScanning":
        obj = cls()
        pos = 0
        while pos < len(data):
            field_num, wire_type, value, pos = _decode_field(data, pos)
            if field_num == 1:
                obj.scanning_state, _ = _decode_varint(value, 0)
            elif field_num == 2:
                obj.scan_id, _ = _decode_varint(value, 0)
            elif field_num == 3:
                obj.total_entries, _ = _decode_varint(value, 0)
            elif field_num == 4:
                obj.total_configured_ssid, _ = _decode_varint(value, 0)
        return obj


@dataclass
class NotifProvisioningState:
    provisioning_state: int = 0

    @classmethod
    def FromString(cls, data: bytes) -> "NotifProvisioningState":
        obj = cls()
        pos = 0
        while pos < len(data):
            field_num, wire_type, value, pos = _decode_field(data, pos)
            if field_num == 1:
                obj.provisioning_state, _ = _decode_varint(value, 0)
        return obj


# COHN Messages
@dataclass
class RequestGetCOHNStatus:
    register_cohn_status: bool = False

    def SerializePartialToString(self) -> bytes:
        if self.register_cohn_status:
            return _encode_bool(1, self.register_cohn_status)
        return b""


@dataclass
class RequestCreateCOHNCert:
    override: bool = False

    def SerializePartialToString(self) -> bytes:
        if self.override:
            return _encode_bool(1, self.override)
        return b""


@dataclass
class RequestClearCOHNCert:
    def SerializePartialToString(self) -> bytes:
        return b""


@dataclass
class RequestCOHNCert:
    def SerializePartialToString(self) -> bytes:
        return b""


@dataclass
class RequestSetCOHNSetting:
    cohn_active: bool = False

    def SerializePartialToString(self) -> bytes:
        return _encode_bool(1, self.cohn_active)


@dataclass
class NotifyCOHNStatus:
    status: int = 0
    state: int = 0
    username: str = ""
    password: str = ""
    ipaddress: str = ""
    enabled: bool = False
    ssid: str = ""
    macaddress: str = ""

    @classmethod
    def FromString(cls, data: bytes) -> "NotifyCOHNStatus":
        obj = cls()
        pos = 0
        while pos < len(data):
            field_num, wire_type, value, pos = _decode_field(data, pos)
            if field_num == 1:
                obj.status, _ = _decode_varint(value, 0)
            elif field_num == 2:
                obj.state, _ = _decode_varint(value, 0)
            elif field_num == 3:
                obj.username = value.decode("utf-8")
            elif field_num == 4:
                obj.password = value.decode("utf-8")
            elif field_num == 5:
                obj.ipaddress = value.decode("utf-8")
            elif field_num == 6:
                obj.enabled, _ = _decode_varint(value, 0)
                obj.enabled = bool(obj.enabled)
            elif field_num == 7:
                obj.ssid = value.decode("utf-8")
            elif field_num == 8:
                obj.macaddress = value.decode("utf-8")
        return obj


@dataclass
class ResponseCOHNCert:
    result: int = 0
    cert: str = ""

    @classmethod
    def FromString(cls, data: bytes) -> "ResponseCOHNCert":
        obj = cls()
        pos = 0
        while pos < len(data):
            field_num, wire_type, value, pos = _decode_field(data, pos)
            if field_num == 1:
                obj.result, _ = _decode_varint(value, 0)
            elif field_num == 2:
                obj.cert = value.decode("utf-8")
        return obj
