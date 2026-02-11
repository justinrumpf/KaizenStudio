"""
Test: Query GoPro battery charge status via COHN.
Hits /gopro/camera/state and reads battery-related status IDs:
  - Status  1: Battery present (0=no, 1=yes)
  - Status  2: Battery bars (0-3) or charging (4)
  - Status 70: Battery percentage (0-100)
"""

import json
import requests

with open("cohn_credentials.json") as f:
    config = json.load(f)

session = requests.Session()
session.verify = "cohn.crt"
session.auth = (config["username"], config["password"])
base = f"https://{config['ip_address']}"

print("=" * 50)
print("GoPro Battery Status Test")
print("=" * 50)

try:
    url = f"{base}/gopro/camera/state"
    print(f"\nRequesting {url} ...")
    r = session.get(url, timeout=10)
    r.raise_for_status()
    state = r.json()

    status = state.get("status", {})

    # Status 1: Battery present
    batt_present = status.get("1")
    # Status 2: Battery bars / charging indicator
    batt_bars = status.get("2")
    # Status 70: Battery percentage
    batt_pct = status.get("70")

    print(f"\n  Battery present : {batt_present}  (1=yes, 0=no)")

    bars_labels = {0: "Empty", 1: "Low", 2: "Medium", 3: "Full", 4: "Charging"}
    bars_label = bars_labels.get(batt_bars, f"Unknown ({batt_bars})")
    print(f"  Battery bars    : {batt_bars}  ({bars_label})")

    print(f"  Battery percent : {batt_pct}%")

    charging = batt_bars == 4
    print(f"\n  Charging?       : {'YES' if charging else 'No'}")

    print("\n" + "=" * 50)
    if batt_pct is not None:
        print(f"SUCCESS - Battery at {batt_pct}%"
              f"{' (charging)' if charging else ''}")
    else:
        print("WARNING - Battery percentage not found in response.")
        print("  Full status keys:", list(status.keys()))
    print("=" * 50)

except requests.exceptions.ConnectionError as e:
    print(f"\nFAILED - Cannot reach camera: {e}")
except requests.exceptions.Timeout:
    print("\nFAILED - Request timed out (camera may be off or unreachable)")
except Exception as e:
    print(f"\nFAILED - {type(e).__name__}: {e}")
