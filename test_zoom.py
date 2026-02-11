"""
Test zoom: properly exit and re-enter webcam mode.
The camera may be stuck — need to fully exit webcam mode first.
"""

import json
import time
import requests

with open("cohn_credentials.json") as f:
    config = json.load(f)

session = requests.Session()
session.verify = "cohn.crt"
session.auth = (config["username"], config["password"])
base = f"https://{config['ip_address']}"


def call(label, endpoint):
    url = f"{base}{endpoint}"
    try:
        r = session.get(url, timeout=10)
        body = r.text[:300] if r.text else "(empty)"
        print(f"  [{r.status_code}] {label}")
        if len(r.text) < 300:
            print(f"         {body}")
        return r.status_code
    except Exception as e:
        print(f"  [ERR]  {label} -> {e}")
        return None


print("=" * 60)
print("GoPro Zoom — Full Reset Test")
print("=" * 60)

# Step 1: Fully reset the camera's webcam state
print("\n--- Step 1: Full webcam reset ---")
call("Webcam stop", "/gopro/webcam/stop")
time.sleep(0.5)
call("Webcam exit", "/gopro/webcam/exit")
time.sleep(2)
call("Webcam status", "/gopro/webcam/status")

# Step 2: Try digital zoom while fully out of webcam mode
print("\n--- Step 2: Digital zoom (fully out of webcam) ---")
call("Zoom 50%", "/gopro/camera/digital_zoom?percent=50")

# Step 3: Enter webcam mode fresh
print("\n--- Step 3: Start webcam fresh ---")
call("Webcam start", "/gopro/webcam/start")
time.sleep(3)
call("Webcam status", "/gopro/webcam/status")

# Step 4: Now do the stop -> zoom -> restart cycle
print("\n--- Step 4: Stop -> zoom 80% -> restart ---")
call("Webcam stop", "/gopro/webcam/stop")
time.sleep(0.5)
call("Zoom 80%", "/gopro/camera/digital_zoom?percent=80")
time.sleep(0.3)
call("Webcam start", "/gopro/webcam/start")
time.sleep(3)
call("Webcam status", "/gopro/webcam/status")

# Step 5: Try zoom with exit instead of stop
print("\n--- Step 5: Exit -> zoom 60% -> start ---")
call("Webcam exit", "/gopro/webcam/exit")
time.sleep(2)
call("Zoom 60%", "/gopro/camera/digital_zoom?percent=60")
time.sleep(0.3)
call("Webcam start", "/gopro/webcam/start")
time.sleep(3)
call("Webcam status", "/gopro/webcam/status")

# Step 6: Try presets - maybe camera needs to be in video mode
print("\n--- Step 6: Try setting video preset first ---")
call("Webcam stop", "/gopro/webcam/stop")
time.sleep(0.5)
call("Webcam exit", "/gopro/webcam/exit")
time.sleep(1)
# Try loading video preset group
call("Set video mode (group 1000)", "/gopro/camera/presets/set_group?id=1000")
time.sleep(1)
call("Zoom 70%", "/gopro/camera/digital_zoom?percent=70")
time.sleep(0.5)
# Check if zoom took effect
call("Webcam start", "/gopro/webcam/start")
time.sleep(3)
call("Webcam status", "/gopro/webcam/status")

# Cleanup
print("\n--- Cleanup ---")
call("Webcam stop", "/gopro/webcam/stop")
call("Webcam exit", "/gopro/webcam/exit")
time.sleep(1)
call("Zoom 0%", "/gopro/camera/digital_zoom?percent=0")

print("\n" + "=" * 60)
print("Look for [200] on zoom AND webcam start calls.")
print("If Step 4 works: stop -> zoom -> start is the way.")
print("If Step 5 works: exit -> zoom -> start is needed.")
print("If Step 6 works: need video preset before zoom.")
print("=" * 60)
