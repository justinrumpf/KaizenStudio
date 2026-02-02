"""
Test GoPro streaming - starts the stream so you can view it in VLC.

VLC steps:
1. Run this script
2. Open VLC -> Media -> Open Network Stream
3. Enter: udp://@:8556
4. Click Play

If that doesn't work, also try:
- udp://0.0.0.0:8556
- udp://@:8554
"""

import json
import time
import requests
from pathlib import Path

CREDS_FILE = Path(__file__).parent / "cohn_credentials.json"
CERT_FILE = Path(__file__).parent / "cohn.crt"

def main():
    # Load credentials
    with open(CREDS_FILE) as f:
        config = json.load(f)

    # Create session
    session = requests.Session()
    session.verify = str(CERT_FILE)
    session.auth = (config["username"], config["password"])

    ip = config["ip_address"]
    base = f"https://{ip}"

    print(f"GoPro IP: {ip}")
    print("-" * 40)

    # Stop any existing streams
    print("Stopping existing streams...")
    try:
        session.get(f"{base}/gopro/webcam/stop", timeout=3)
    except:
        pass
    try:
        session.get(f"{base}/gopro/camera/stream/stop", timeout=3)
    except:
        pass
    time.sleep(1)

    # Set video mode
    print("Setting video mode...")
    try:
        session.get(f"{base}/gopro/camera/presets/set_group?id=1000", timeout=5)
        time.sleep(1)
    except Exception as e:
        print(f"  Warning: {e}")

    # Try preview stream first
    print("\n=== Trying Preview Stream ===")
    try:
        r = session.get(f"{base}/gopro/camera/stream/start", timeout=10)
        r.raise_for_status()
        print("Preview stream started!")
        print("\nOpen VLC -> Media -> Open Network Stream")
        print("Enter: udp://@:8556")
        print("\nPress Enter to stop...")
        input()
        session.get(f"{base}/gopro/camera/stream/stop", timeout=5)
        return
    except requests.exceptions.HTTPError as e:
        print(f"Preview stream failed: {e.response.status_code}")

    # Try webcam mode
    print("\n=== Trying Webcam Mode ===")
    try:
        r = session.get(f"{base}/gopro/webcam/start", timeout=10)
        r.raise_for_status()
        print("Webcam started!")

        # Check status
        r = session.get(f"{base}/gopro/webcam/status", timeout=5)
        status = r.json()
        print(f"Webcam status: {status}")

        print("\nTry in VLC:")
        print("  udp://@:8556")
        print("  udp://@:8554")
        print(f"  rtsp://{ip}:8554/live")

        print("\nPress Enter to stop...")
        input()
        session.get(f"{base}/gopro/webcam/stop", timeout=5)
    except Exception as e:
        print(f"Webcam failed: {e}")

if __name__ == "__main__":
    main()
