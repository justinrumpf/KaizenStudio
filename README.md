# Kaizen Studio

AI-powered background removal studio with GoPro COHN integration for e-commerce product photography.

## Prerequisites

- **Python 3.9+** — [python.org/downloads](https://www.python.org/downloads/)
  - Make sure "Add Python to PATH" is checked during install
- **Node.js 14+** — [nodejs.org](https://nodejs.org/) (only needed for `npm` script runner)
- **FFmpeg** — Required for GoPro live preview streaming
  - Install via `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - Must be in your system PATH
- **Git** — Required to install MobileSAM from GitHub

## Quick Start

```bash
# 1. Clone/copy the project, then cd into it
cd KaizenStudio

# 2. Run full setup (creates venv, installs all Python packages)
npm run setup

# 3. Download MobileSAM weights (~39 MB) into the weights/ folder
#    The setup script will print the URL if the file is missing.
#    Direct link:
curl -L -o weights/mobile_sam.pt https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

# 4. Start the app
npm start
```

The app will be available at **http://localhost:5000**

## npm Scripts

| Command | What it does |
|---|---|
| `npm run setup` | Creates Python venv, installs all pip packages, checks for model weights |
| `npm run pip-install` | Re-installs Python packages into the existing venv |
| `npm start` | Starts the app using the venv Python |
| `npm run dev` | Starts with Flask debug mode (auto-reload on code changes) |

## Manual Setup (without npm)

If you prefer not to use npm:

```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install Python packages
pip install -r requirements.txt

# Download MobileSAM weights
mkdir weights
curl -L -o weights/mobile_sam.pt https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

# Run
python app.py
```

## GoPro Setup (Optional)

The app supports GoPro cameras over COHN (Camera On Home Network) for live preview and capture.

1. Start the app and click the **Settings** gear icon
2. Click **Provision GoPro** — this pairs with your GoPro over USB and generates `cohn_credentials.json` and `cohn.crt`
3. Once provisioned, the GoPro connects over your local WiFi network automatically

If you're moving to a new machine, you can copy `cohn_credentials.json` and `cohn.crt` from a previously provisioned setup — no need to re-provision.

## Project Structure

```
KaizenStudio/
  app.py                  # Flask backend — all routes and image processing
  stream_manager.py       # GoPro live stream manager (FFmpeg)
  requirements.txt        # Python dependencies
  package.json            # npm scripts for setup/start
  run.bat                 # Legacy Windows launcher
  templates/
    index.html            # Single-page UI (HTML + CSS + JS)
  static/
    logo.png              # Watermark logo
  weights/
    mobile_sam.pt          # MobileSAM model checkpoint (~39 MB)
  uploads/                # Captured/uploaded originals (auto-created)
  outputs/                # Processed results (auto-created)
  venv/                   # Python virtual environment (auto-created)
```

## Troubleshooting

- **"venv\Scripts\python is not recognized"** — Run `npm run setup` first to create the virtual environment.
- **MobileSAM fails to load** — Make sure `weights/mobile_sam.pt` exists. See step 3 in Quick Start.
- **GoPro not connecting** — Ensure your GoPro is on the same WiFi network and COHN is enabled. Re-provision from Settings if needed.
- **FFmpeg not found** — The GoPro live preview requires FFmpeg in your PATH. Install it and restart your terminal.
