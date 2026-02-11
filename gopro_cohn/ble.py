# GoPro BLE Communication Module
# Based on Open GoPro, Version 2.0 (C) Copyright 2021 GoPro, Inc.

from __future__ import annotations
import asyncio
import enum
import re
import logging
from typing import Callable, Awaitable, TypeVar, Generator, Final

from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
from bleak.backends.device import BLEDevice

logger = logging.getLogger("gopro_cohn")

GOPRO_BASE_UUID = "b5f9{}-aa8d-11e3-9046-0002a5d5c51b"

T = TypeVar("T")


class GoProUuid(str, enum.Enum):
    """GoPro BLE UUIDs"""
    COMMAND_REQ_UUID = GOPRO_BASE_UUID.format("0072")
    COMMAND_RSP_UUID = GOPRO_BASE_UUID.format("0073")
    SETTINGS_REQ_UUID = GOPRO_BASE_UUID.format("0074")
    SETTINGS_RSP_UUID = GOPRO_BASE_UUID.format("0075")
    QUERY_REQ_UUID = GOPRO_BASE_UUID.format("0076")
    QUERY_RSP_UUID = GOPRO_BASE_UUID.format("0077")
    NETWORK_MANAGEMENT_REQ_UUID = GOPRO_BASE_UUID.format("0091")
    NETWORK_MANAGEMENT_RSP_UUID = GOPRO_BASE_UUID.format("0092")

    @classmethod
    def dict_by_uuid(cls, value_creator: Callable[[GoProUuid], T]) -> dict[GoProUuid, T]:
        return {uuid: value_creator(uuid) for uuid in cls}


NotificationHandler = Callable[[BleakGATTCharacteristic, bytearray], Awaitable[None]]


async def connect_ble(
    notification_handler: NotificationHandler,
    identifier: str | None = None,
    timeout: int = 10,
    max_retries: int = 5
) -> BleakClient:
    """
    Connect to a GoPro camera via BLE.

    Args:
        notification_handler: Async callback for BLE notifications
        identifier: Last 4 digits of GoPro serial number (optional)
        timeout: Scan timeout in seconds
        max_retries: Maximum connection retry attempts

    Returns:
        Connected BleakClient instance
    """
    gopro_device: BLEDevice | None = None
    event = asyncio.Event()

    # Build device name pattern
    if identifier:
        pattern = re.compile(f"GoPro {identifier}")
    else:
        pattern = re.compile(r"GoPro [A-Z0-9]{4}")

    def scan_callback(device: BLEDevice, _) -> None:
        nonlocal gopro_device
        if device.name and pattern.match(device.name):
            gopro_device = device
            event.set()

    for retry in range(max_retries):
        logger.info(f"Scanning for GoPro (attempt {retry + 1}/{max_retries})...")
        event.clear()
        gopro_device = None

        scanner = BleakScanner(detection_callback=scan_callback)
        await scanner.start()

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Scan timeout, retrying...")
            continue
        finally:
            await scanner.stop()

        if gopro_device:
            break

    if not gopro_device:
        raise RuntimeError("Could not find a GoPro device. Ensure it's powered on and in pairing mode.")

    logger.info(f"Found GoPro: {gopro_device.name} ({gopro_device.address})")

    # Connect to the device
    for retry in range(max_retries):
        try:
            logger.info(f"Connecting to {gopro_device.name} (attempt {retry + 1}/{max_retries})...")
            client = BleakClient(gopro_device, timeout=15)
            await client.connect()

            logger.info("Connected! Attempting to pair...")
            try:
                await client.pair()
                logger.info("Pairing successful")
            except NotImplementedError:
                # Pairing may not be supported on all platforms
                logger.debug("Pairing not supported on this platform")

            # Enable notifications only on GoPro-specific characteristics
            logger.info("Enabling notifications...")
            gopro_uuids = {uuid.value for uuid in GoProUuid}
            for service in client.services:
                for char in service.characteristics:
                    if "notify" in char.properties and char.uuid in gopro_uuids:
                        try:
                            await client.start_notify(char, notification_handler)
                            logger.debug(f"Enabled notifications for {char.uuid}")
                        except Exception as e:
                            logger.debug(f"Could not enable notifications for {char.uuid}: {e}")

            logger.info("BLE connection established successfully")
            return client

        except Exception as e:
            logger.warning(f"Connection failed: {e}")
            if retry < max_retries - 1:
                await asyncio.sleep(2)
            else:
                raise RuntimeError(f"Failed to connect after {max_retries} attempts: {e}")

    raise RuntimeError("Failed to connect to GoPro")


def yield_fragmented_packets(payload: bytes) -> Generator[bytes, None, None]:
    """
    Fragment a payload into BLE-sized packets (max 20 bytes).

    Args:
        payload: The payload to fragment

    Yields:
        Fragmented packet bytes
    """
    length = len(payload)
    CONTINUATION_HEADER: Final = bytearray([0x80])
    MAX_PACKET_SIZE: Final = 20
    is_first_packet = True

    # Build initial length header
    if length < (2**13 - 1):
        header = bytearray((length | 0x2000).to_bytes(2, "big", signed=False))
    elif length < (2**16 - 1):
        header = bytearray((length | 0x6400).to_bytes(2, "big", signed=False))
    else:
        raise ValueError(f"Data length {length} is too large for BLE protocol")

    byte_index = 0
    while bytes_remaining := length - byte_index:
        if is_first_packet:
            packet = bytearray(header)
            is_first_packet = False
        else:
            packet = bytearray(CONTINUATION_HEADER)

        packet_size = min(MAX_PACKET_SIZE - len(packet), bytes_remaining)
        packet.extend(bytearray(payload[byte_index : byte_index + packet_size]))
        yield bytes(packet)
        byte_index += packet_size


async def fragment_and_write(client: BleakClient, uuid: str, data: bytes) -> None:
    """Fragment data and write to a BLE characteristic."""
    for packet in yield_fragmented_packets(data):
        await client.write_gatt_char(uuid, packet, response=True)
