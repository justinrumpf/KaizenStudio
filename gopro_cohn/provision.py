# GoPro COHN Provisioning
# Based on Open GoPro, Version 2.0 (C) Copyright 2021 GoPro, Inc.

from __future__ import annotations
import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import cast

import pytz
from tzlocal import get_localzone

from .ble import GoProUuid, connect_ble, fragment_and_write
from .responses import ResponseManager
from . import proto

logger = logging.getLogger("gopro_cohn")


@dataclass(frozen=True)
class Credentials:
    """COHN credentials for HTTPS communication with the camera."""
    certificate: str
    username: str
    password: str
    ip_address: str

    def __str__(self) -> str:
        return json.dumps(asdict(self), indent=4)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, cert_path: Path, creds_path: Path | None = None) -> None:
        """Save certificate and optionally credentials to files."""
        with open(cert_path, "w") as f:
            f.write(self.certificate)
        logger.info(f"Certificate saved to {cert_path}")

        if creds_path:
            with open(creds_path, "w") as f:
                json.dump({
                    "username": self.username,
                    "password": self.password,
                    "ip_address": self.ip_address,
                    "certificate_path": str(cert_path)
                }, f, indent=2)
            logger.info(f"Credentials saved to {creds_path}")


async def _set_date_time(manager: ResponseManager) -> None:
    """Synchronize camera date/time with local system."""
    tz = pytz.timezone(get_localzone().key)
    now = tz.localize(datetime.now(), is_dst=None)

    try:
        is_dst = now.tzinfo._dst.seconds != 0  # type: ignore
        offset = (now.utcoffset().total_seconds() - now.tzinfo._dst.seconds) / 60  # type: ignore
    except AttributeError:
        is_dst = False
        offset = now.utcoffset().total_seconds() / 60  # type: ignore

    if is_dst:
        offset += 60
    offset = int(offset)

    logger.info(f"Setting camera date/time to {now} (offset={offset}, DST={is_dst})")

    request = bytearray([
        0x0F,  # Command ID
        10,    # Length
        *now.year.to_bytes(2, "big", signed=False),
        now.month,
        now.day,
        now.hour,
        now.minute,
        now.second,
        *offset.to_bytes(2, "big", signed=True),
        is_dst,
    ])
    request.insert(0, len(request))

    await manager.client.write_gatt_char(GoProUuid.COMMAND_REQ_UUID.value, request, response=True)
    response = await manager.get_next_tlv()
    assert response.id == 0x0F and response.status == 0x00
    logger.info("Date/time set successfully")


async def _scan_for_networks(manager: ResponseManager) -> int:
    """Scan for WiFi networks and return scan ID."""
    logger.info("Scanning for WiFi networks...")

    request = bytearray([
        0x02,  # Feature ID
        0x02,  # Action ID
        *proto.RequestStartScan().SerializePartialToString(),
    ])
    request.insert(0, len(request))

    await manager.client.write_gatt_char(
        GoProUuid.NETWORK_MANAGEMENT_REQ_UUID.value, request, response=True
    )

    while True:
        response = await manager.get_next_protobuf()
        if response.feature_id != 0x02:
            raise RuntimeError(f"Unexpected feature ID: {response.feature_id}")

        if response.action_id == 0x82:  # Initial response
            manager.assert_success(response.data)
        elif response.action_id == 0x0B:  # Scan notification
            notification = cast(proto.NotifStartScanning, response.data)
            logger.debug(f"Scan state: {notification.scanning_state}")
            if notification.scanning_state == proto.EnumScanning.SCANNING_SUCCESS:
                logger.info(f"Scan complete, found {notification.total_entries} networks")
                return notification.scan_id
        else:
            raise RuntimeError(f"Unexpected action ID: {response.action_id}")


async def _get_scan_results(manager: ResponseManager, scan_id: int) -> list:
    """Retrieve WiFi scan results."""
    logger.info("Retrieving scan results...")

    request = bytearray([
        0x02,  # Feature ID
        0x03,  # Action ID
        *proto.RequestGetApEntries(
            start_index=0, max_entries=100, scan_id=scan_id
        ).SerializePartialToString(),
    ])
    request.insert(0, len(request))

    await manager.client.write_gatt_char(
        GoProUuid.NETWORK_MANAGEMENT_REQ_UUID.value, request, response=True
    )

    response = await manager.get_next_protobuf()
    if response.feature_id != 0x02 or response.action_id != 0x83:
        raise RuntimeError("Unexpected response to scan results request")

    entries = cast(proto.ResponseGetApEntries, response.data)
    manager.assert_success(entries)

    for entry in entries.entries:
        logger.info(f"  Found: {entry.ssid} (signal: {entry.signal_strength_bars} bars)")

    return list(entries.entries)


async def _connect_to_network(manager: ResponseManager, ssid: str, password: str) -> None:
    """Connect to a WiFi network."""
    logger.info(f"Connecting to WiFi network: {ssid}")

    # Get scan results to find the network
    scan_id = await _scan_for_networks(manager)
    entries = await _get_scan_results(manager, scan_id)

    target_entry = None
    for entry in entries:
        if entry.ssid == ssid:
            target_entry = entry
            break

    if not target_entry:
        raise RuntimeError(f"Network '{ssid}' not found in scan results")

    # Check if already configured
    if target_entry.scan_entry_flags & proto.EnumScanEntryFlags.SCAN_FLAG_CONFIGURED:
        logger.info("Network previously configured, reconnecting...")
        request = bytearray([
            0x02,  # Feature ID
            0x04,  # Action ID (RequestConnect)
            *proto.RequestConnect(ssid=ssid).SerializePartialToString(),
        ])
        expected_action_id = 0x84
    else:
        logger.info("Connecting to new network...")
        request = bytearray([
            0x02,  # Feature ID
            0x05,  # Action ID (RequestConnectNew)
            *proto.RequestConnectNew(ssid=ssid, password=password).SerializePartialToString(),
        ])
        expected_action_id = 0x85

    await fragment_and_write(
        manager.client, GoProUuid.NETWORK_MANAGEMENT_REQ_UUID.value, request
    )

    while True:
        response = await manager.get_next_protobuf()
        if response.feature_id != 0x02:
            raise RuntimeError(f"Unexpected feature ID: {response.feature_id}")

        if response.action_id in (0x84, 0x85):
            manager.assert_success(response.data)
        elif response.action_id == 0x0C:  # Provisioning notification
            notification = cast(proto.NotifProvisioningState, response.data)
            logger.debug(f"Provisioning state: {notification.provisioning_state}")
            if notification.provisioning_state in (
                proto.EnumProvisioning.PROVISIONING_SUCCESS_NEW_AP,
                proto.EnumProvisioning.PROVISIONING_SUCCESS_OLD_AP,
            ):
                logger.info(f"Successfully connected to {ssid}")
                return
            elif notification.provisioning_state == proto.EnumProvisioning.PROVISIONING_ERROR_PASSWORD_AUTH:
                raise RuntimeError("WiFi authentication failed - check password")
            elif notification.provisioning_state == proto.EnumProvisioning.PROVISIONING_ERROR_FAILED_TO_ASSOCIATE:
                raise RuntimeError("Failed to associate with WiFi network")
            elif notification.provisioning_state not in (
                proto.EnumProvisioning.PROVISIONING_STARTED,
                proto.EnumProvisioning.PROVISIONING_NEVER_STARTED,
            ):
                raise RuntimeError(f"Provisioning error: {notification.provisioning_state}")
        else:
            raise RuntimeError(f"Unexpected action ID: {response.action_id}")


async def _clear_certificate(manager: ResponseManager) -> None:
    """Clear existing COHN certificate."""
    logger.info("Clearing existing COHN certificate...")

    request = bytearray([
        0xF1,  # Feature ID
        0x66,  # Action ID
        *proto.RequestClearCOHNCert().SerializePartialToString(),
    ])
    request.insert(0, len(request))

    await manager.client.write_gatt_char(
        GoProUuid.COMMAND_REQ_UUID.value, request, response=True
    )

    response = await manager.get_next_protobuf()
    if response.feature_id != 0xF1 or response.action_id != 0xE6:
        raise RuntimeError("Unexpected response to clear certificate request")

    manager.assert_success(response.data)
    logger.info("Certificate cleared")


async def _create_certificate(manager: ResponseManager) -> None:
    """Create new COHN certificate."""
    logger.info("Creating new COHN certificate...")

    request = bytearray([
        0xF1,  # Feature ID
        0x67,  # Action ID
        *proto.RequestCreateCOHNCert().SerializePartialToString(),
    ])
    request.insert(0, len(request))

    await manager.client.write_gatt_char(
        GoProUuid.COMMAND_REQ_UUID.value, request, response=True
    )

    response = await manager.get_next_protobuf()
    if response.feature_id != 0xF1 or response.action_id != 0xE7:
        raise RuntimeError("Unexpected response to create certificate request")

    manager.assert_success(response.data)
    logger.info("Certificate created")


async def _get_certificate(manager: ResponseManager) -> str:
    """Retrieve COHN certificate."""
    logger.info("Retrieving COHN certificate...")

    request = bytearray([
        0xF5,  # Feature ID
        0x6E,  # Action ID
        *proto.RequestCOHNCert().SerializePartialToString(),
    ])
    request.insert(0, len(request))

    await manager.client.write_gatt_char(
        GoProUuid.QUERY_REQ_UUID.value, request, response=True
    )

    response = await manager.get_next_protobuf()
    if response.feature_id != 0xF5 or response.action_id != 0xEE:
        raise RuntimeError("Unexpected response to get certificate request")

    cert_response = cast(proto.ResponseCOHNCert, response.data)
    manager.assert_success(cert_response)
    logger.info("Certificate retrieved")
    return cert_response.cert


async def _get_cohn_status(manager: ResponseManager) -> proto.NotifyCOHNStatus:
    """Wait for COHN to be fully provisioned and return status."""
    logger.info("Waiting for COHN provisioning to complete...")

    request = bytearray([
        0xF5,  # Feature ID
        0x6F,  # Action ID
        *proto.RequestGetCOHNStatus(register_cohn_status=True).SerializePartialToString(),
    ])
    request.insert(0, len(request))

    await manager.client.write_gatt_char(
        GoProUuid.QUERY_REQ_UUID.value, request, response=True
    )

    while True:
        response = await manager.get_next_protobuf()
        if response.feature_id != 0xF5 or response.action_id != 0xEF:
            raise RuntimeError("Unexpected response to COHN status request")

        status = cast(proto.NotifyCOHNStatus, response.data)
        logger.debug(f"COHN state: {status.state}, user={status.username!r}, ip={status.ipaddress!r}")

        if status.state == proto.EnumCOHNNetworkState.COHN_STATE_NetworkConnected:
            # Wait until credentials are fully populated (new cameras may
            # send NetworkConnected before username/password are ready)
            if status.username and status.password and status.ipaddress:
                logger.info(f"COHN connected! IP: {status.ipaddress}")
                return status
            else:
                logger.info("COHN connected but credentials not yet ready, waiting...")
        elif status.state == proto.EnumCOHNNetworkState.COHN_STATE_Error:
            raise RuntimeError("COHN encountered an error")


async def provision_gopro(
    ssid: str,
    password: str,
    identifier: str | None = None,
    cert_path: Path | None = None,
    creds_path: Path | None = None,
) -> Credentials:
    """
    Provision a GoPro camera for COHN (Camera on Home Network).

    This connects to the GoPro via Bluetooth, configures it to connect to your
    WiFi network, and retrieves the credentials needed for HTTPS communication.

    Args:
        ssid: WiFi network SSID
        password: WiFi network password
        identifier: Last 4 digits of GoPro serial number (optional)
        cert_path: Path to save the SSL certificate (optional)
        creds_path: Path to save credentials JSON (optional)

    Returns:
        Credentials object containing certificate, username, password, and IP
    """
    manager = ResponseManager()
    credentials: Credentials | None = None

    try:
        # Connect via BLE
        client = await connect_ble(manager.notification_handler, identifier)
        manager.set_client(client)

        # Set camera date/time (required for valid certificate)
        await _set_date_time(manager)

        # Connect to WiFi network
        await _connect_to_network(manager, ssid, password)

        # Provision COHN
        logger.info("Provisioning COHN...")
        await _clear_certificate(manager)
        await _create_certificate(manager)
        certificate = await _get_certificate(manager)
        status = await _get_cohn_status(manager)

        credentials = Credentials(
            certificate=certificate,
            username=status.username,
            password=status.password,
            ip_address=status.ipaddress,
        )

        logger.info("COHN provisioning complete!")
        logger.info(f"  IP Address: {credentials.ip_address}")
        logger.info(f"  Username: {credentials.username}")

        # Save credentials if paths provided
        if cert_path:
            credentials.save(cert_path, creds_path)

    finally:
        if manager.is_initialized:
            await manager.client.disconnect()
            logger.info("Disconnected from GoPro")

    if credentials is None:
        raise RuntimeError("Provisioning failed")

    return credentials
