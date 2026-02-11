# GoPro BLE Response Handling
# Based on Open GoPro, Version 2.0 (C) Copyright 2021 GoPro, Inc.

from __future__ import annotations
import asyncio
import enum
import logging
from dataclasses import dataclass
from typing import cast, Any, Type

from bleak import BleakClient
from bleak.backends.characteristic import BleakGATTCharacteristic

from .ble import GoProUuid
from . import proto

logger = logging.getLogger("gopro_cohn")


class Response:
    """Base class for accumulating BLE responses."""

    def __init__(self, uuid: GoProUuid) -> None:
        self.bytes_remaining = 0
        self.uuid = uuid
        self.raw_bytes = bytearray()

    @classmethod
    def from_received(cls, received: Response):
        response = cls(received.uuid)
        response.bytes_remaining = 0
        response.raw_bytes = received.raw_bytes
        return response

    @property
    def is_received(self) -> bool:
        return len(self.raw_bytes) > 0 and self.bytes_remaining == 0

    def accumulate(self, data: bytes) -> None:
        """Accumulate incoming BLE packet data."""
        CONT_MASK = 0b10000000
        HDR_MASK = 0b01100000
        GEN_LEN_MASK = 0b00011111
        EXT_13_BYTE0_MASK = 0b00011111

        class Header(enum.Enum):
            GENERAL = 0b00
            EXT_13 = 0b01
            EXT_16 = 0b10
            RESERVED = 0b11

        buf = bytearray(data)
        if buf[0] & CONT_MASK:
            buf.pop(0)
        else:
            self.raw_bytes = bytearray()
            hdr = Header((buf[0] & HDR_MASK) >> 5)
            if hdr is Header.GENERAL:
                self.bytes_remaining = buf[0] & GEN_LEN_MASK
                buf = buf[1:]
            elif hdr is Header.EXT_13:
                self.bytes_remaining = ((buf[0] & EXT_13_BYTE0_MASK) << 8) + buf[1]
                buf = buf[2:]
            elif hdr is Header.EXT_16:
                self.bytes_remaining = (buf[1] << 8) + buf[2]
                buf = buf[3:]

        self.raw_bytes.extend(buf)
        self.bytes_remaining -= len(buf)


class TlvResponse(Response):
    """TLV (Type-Length-Value) response."""

    def __init__(self, uuid: GoProUuid) -> None:
        super().__init__(uuid)
        self.id: int = 0
        self.status: int = 0
        self.payload: bytes = b""

    def parse(self) -> None:
        self.id = self.raw_bytes[0]
        self.status = self.raw_bytes[1]
        self.payload = bytes(self.raw_bytes[2:])


class ProtobufResponse(Response):
    """Protobuf response."""

    def __init__(self, uuid: GoProUuid) -> None:
        super().__init__(uuid)
        self.feature_id: int = 0
        self.action_id: int = 0
        self.data: Any = None

    def parse(self, proto_message: Type[Any]) -> None:
        self.feature_id = self.raw_bytes[0]
        self.action_id = self.raw_bytes[1]
        self.data = proto_message.FromString(bytes(self.raw_bytes[2:]))


@dataclass(frozen=True)
class ProtobufId:
    """Protobuf feature/action ID pair."""
    feature_id: int
    action_id: int


# Mapping of protobuf IDs to message types
PROTOBUF_ID_TO_MESSAGE: dict[ProtobufId, Type[Any] | None] = {
    # Network management
    ProtobufId(0x02, 0x02): None,
    ProtobufId(0x02, 0x04): None,
    ProtobufId(0x02, 0x05): None,
    ProtobufId(0x02, 0x0B): proto.NotifStartScanning,
    ProtobufId(0x02, 0x0C): proto.NotifProvisioningState,
    ProtobufId(0x02, 0x82): proto.ResponseStartScanning,
    ProtobufId(0x02, 0x83): proto.ResponseGetApEntries,
    ProtobufId(0x02, 0x84): proto.ResponseConnect,
    ProtobufId(0x02, 0x85): proto.ResponseConnectNew,
    # COHN
    ProtobufId(0xF1, 0x66): None,
    ProtobufId(0xF1, 0x67): None,
    ProtobufId(0xF1, 0xE6): proto.ResponseGeneric,
    ProtobufId(0xF1, 0xE7): proto.ResponseGeneric,
    ProtobufId(0xF5, 0x6E): None,
    ProtobufId(0xF5, 0x6F): None,
    ProtobufId(0xF5, 0xEE): proto.ResponseCOHNCert,
    ProtobufId(0xF5, 0xEF): proto.NotifyCOHNStatus,
}


class ResponseManager:
    """Manages BLE response accumulation and parsing."""

    def __init__(self) -> None:
        self._responses_by_uuid = GoProUuid.dict_by_uuid(Response)
        self._queue: asyncio.Queue[ProtobufResponse | TlvResponse] = asyncio.Queue()
        self._client: BleakClient | None = None

    def set_client(self, client: BleakClient) -> None:
        self._client = client

    @property
    def is_initialized(self) -> bool:
        return self._client is not None

    @property
    def client(self) -> BleakClient:
        if not self.is_initialized:
            raise RuntimeError("Client has not been set")
        return self._client  # type: ignore

    def _decipher_response(self, response: Response) -> ProtobufResponse | TlvResponse:
        """Parse a raw response into the appropriate type."""
        payload = response.raw_bytes
        proto_id = ProtobufId(payload[0], payload[1])

        if proto_id in PROTOBUF_ID_TO_MESSAGE:
            proto_message = PROTOBUF_ID_TO_MESSAGE.get(proto_id)
            if proto_message is None:
                # Use ResponseGeneric as fallback
                proto_message = proto.ResponseGeneric

            parsed = ProtobufResponse.from_received(response)
            parsed.parse(proto_message)
            return parsed
        else:
            # TLV response
            parsed = TlvResponse.from_received(response)
            parsed.parse()
            return parsed

    async def notification_handler(
        self, characteristic: BleakGATTCharacteristic, data: bytearray
    ) -> None:
        """Handle incoming BLE notifications."""
        uuid = GoProUuid(self.client.services.characteristics[characteristic.handle].uuid)
        logger.debug(f"Received at {uuid}: {data.hex(':')}")

        response = self._responses_by_uuid[uuid]
        response.accumulate(data)

        if response.is_received:
            await self._queue.put(self._decipher_response(response))
            self._responses_by_uuid[uuid] = Response(uuid)

    @staticmethod
    def assert_success(response: Any) -> None:
        """Assert a ResponseGeneric indicates success."""
        if hasattr(response, "result"):
            result = int(response.result)
            if result != proto.EnumResultGeneric.RESULT_SUCCESS:
                raise RuntimeError(f"Request failed with result: {result}")

    async def get_next_response(self) -> ProtobufResponse | TlvResponse:
        return await self._queue.get()

    async def get_next_tlv(self) -> TlvResponse:
        return cast(TlvResponse, await self.get_next_response())

    async def get_next_protobuf(self) -> ProtobufResponse:
        return cast(ProtobufResponse, await self.get_next_response())
