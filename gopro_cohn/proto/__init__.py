# GoPro Protobuf Messages
# Manual implementation to avoid protobuf compilation requirements

from .messages import (
    # Enums
    EnumResultGeneric,
    EnumCOHNStatus,
    EnumCOHNNetworkState,
    EnumProvisioning,
    EnumScanning,
    EnumScanEntryFlags,
    # Response Generic
    ResponseGeneric,
    # Network Management
    RequestStartScan,
    RequestGetApEntries,
    RequestConnect,
    RequestConnectNew,
    ResponseGetApEntries,
    ResponseConnect,
    ResponseConnectNew,
    ResponseStartScanning,
    NotifStartScanning,
    NotifProvisioningState,
    ScanEntry,
    # COHN
    RequestGetCOHNStatus,
    RequestCreateCOHNCert,
    RequestClearCOHNCert,
    RequestCOHNCert,
    RequestSetCOHNSetting,
    NotifyCOHNStatus,
    ResponseCOHNCert,
)

__all__ = [
    "EnumResultGeneric",
    "EnumCOHNStatus",
    "EnumCOHNNetworkState",
    "EnumProvisioning",
    "EnumScanning",
    "EnumScanEntryFlags",
    "ResponseGeneric",
    "RequestStartScan",
    "RequestGetApEntries",
    "RequestConnect",
    "RequestConnectNew",
    "ResponseGetApEntries",
    "ResponseConnect",
    "ResponseConnectNew",
    "ResponseStartScanning",
    "NotifStartScanning",
    "NotifProvisioningState",
    "ScanEntry",
    "RequestGetCOHNStatus",
    "RequestCreateCOHNCert",
    "RequestClearCOHNCert",
    "RequestCOHNCert",
    "RequestSetCOHNSetting",
    "NotifyCOHNStatus",
    "ResponseCOHNCert",
]
