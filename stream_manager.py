"""
GoPro real-time preview stream manager.

Uses GoPro webcam mode which streams MPEG-TS over UDP port 8554.
"""

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional
import requests


class StreamManager:
    """Manages GoPro live streaming via webcam mode."""

    def __init__(self, creds_file: Path, cert_file: Path, snapshot_dir: Path):
        self.creds_file = creds_file
        self.cert_file = cert_file
        self.snapshot_dir = snapshot_dir

        self.session: Optional[requests.Session] = None
        self.config: Optional[dict] = None

        self.streaming = False
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.frame_thread: Optional[threading.Thread] = None

        self.frame_lock = threading.Lock()
        self.current_frame: Optional[bytes] = None
        self.frame_count = 0

        self.status = "stopped"
        self.error_message = ""

        self.ffmpeg_path = shutil.which('ffmpeg')

    def _init_session(self) -> bool:
        if not self.creds_file.exists() or not self.cert_file.exists():
            self.error_message = "GoPro credentials not found"
            return False

        try:
            with open(self.creds_file) as f:
                self.config = json.load(f)

            self.session = requests.Session()
            self.session.verify = str(self.cert_file)
            self.session.auth = (self.config["username"], self.config["password"])

            adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1)
            self.session.mount("https://", adapter)
            return True
        except Exception as e:
            self.error_message = str(e)
            return False

    def _gopro_request(self, endpoint: str, timeout: int = 10) -> dict:
        if self.session is None:
            if not self._init_session():
                raise Exception(self.error_message)

        url = f"https://{self.config['ip_address']}{endpoint}"
        response = self.session.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json() if response.content else {}

    def start_stream(self) -> bool:
        if self.streaming:
            return True

        self.status = "starting"
        self.error_message = ""
        self.frame_count = 0

        if not self.ffmpeg_path:
            self.error_message = "FFmpeg not found in PATH"
            self.status = "error"
            return False

        try:
            if self.session is None and not self._init_session():
                self.status = "error"
                return False

            # Stop any existing webcam/stream
            print("Cleaning up...")
            try:
                self._gopro_request("/gopro/webcam/stop", timeout=3)
            except:
                pass
            try:
                self._gopro_request("/gopro/camera/stream/stop", timeout=3)
            except:
                pass
            time.sleep(0.5)

            # Start FFmpeg listener FIRST on port 8554
            print("Starting FFmpeg on UDP port 8554...")
            self._start_ffmpeg()

            # Start webcam mode
            print("Starting webcam mode...")
            result = self._gopro_request("/gopro/webcam/start", timeout=10)
            print(f"Webcam result: {result}")

            time.sleep(1)

            # Check status
            status = self._gopro_request("/gopro/webcam/status", timeout=5)
            print(f"Webcam status: {status}")

            # Wait for frames
            print("Waiting for frames...")
            for i in range(30):  # Wait up to 3 seconds
                time.sleep(0.1)
                if self.frame_count > 0:
                    break

            if self.frame_count > 0:
                self.streaming = True
                self.status = "live"
                print(f"Stream is live! Frames: {self.frame_count}")
                return True
            else:
                print("No frames received yet, but stream may be starting...")
                self.streaming = True
                self.status = "live"
                return True

        except Exception as e:
            self.error_message = str(e)
            self.status = "error"
            print(f"Stream error: {e}")
            self._stop_ffmpeg()
            return False

    def _start_ffmpeg(self):
        # Webcam streams on UDP port 8554!
        cmd = [
            self.ffmpeg_path,
            '-hide_banner',
            '-loglevel', 'info',
            '-probesize', '5000000',
            '-analyzeduration', '5000000',
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-i', 'udp://0.0.0.0:8554?overrun_nonfatal=1&fifo_size=50000000',
            '-map', '0:v:0',
            '-an',
            '-f', 'mjpeg',
            '-q:v', '5',
            'pipe:1'
        ]

        print(f"FFmpeg: {' '.join(cmd)}")

        startupinfo = None
        if hasattr(subprocess, 'STARTUPINFO'):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

        self.ffmpeg_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**6,
            startupinfo=startupinfo
        )

        self.frame_thread = threading.Thread(target=self._read_frames, daemon=True)
        self.frame_thread.start()

        threading.Thread(target=self._monitor_stderr, daemon=True).start()

    def _monitor_stderr(self):
        if not self.ffmpeg_process:
            return
        try:
            for line in iter(self.ffmpeg_process.stderr.readline, b''):
                if not self.streaming and self.status != "starting":
                    break
                text = line.decode('utf-8', errors='ignore').strip()
                if text:
                    print(f"FFmpeg: {text}")
        except:
            pass

    def _read_frames(self):
        buffer = b''
        jpeg_start = b'\xff\xd8'
        jpeg_end = b'\xff\xd9'

        print("Frame reader started...")

        while self.ffmpeg_process and (self.streaming or self.status == "starting"):
            try:
                chunk = self.ffmpeg_process.stdout.read(8192)
                if not chunk:
                    break

                buffer += chunk

                while True:
                    start = buffer.find(jpeg_start)
                    if start == -1:
                        buffer = b''
                        break
                    if start > 0:
                        buffer = buffer[start:]

                    end = buffer.find(jpeg_end, 2)
                    if end == -1:
                        break

                    frame = buffer[:end + 2]
                    buffer = buffer[end + 2:]

                    with self.frame_lock:
                        self.current_frame = frame
                        self.frame_count += 1

            except Exception as e:
                print(f"Frame error: {e}")
                break

        print(f"Frame reader stopped. Total: {self.frame_count}")

    def _stop_ffmpeg(self):
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=2)
            except:
                try:
                    self.ffmpeg_process.kill()
                except:
                    pass
            self.ffmpeg_process = None

    def stop_stream(self, keep_error=False):
        self.streaming = False

        self._stop_ffmpeg()

        try:
            self._gopro_request("/gopro/webcam/stop", timeout=5)
        except:
            pass

        with self.frame_lock:
            self.current_frame = None

        if not keep_error:
            self.status = "stopped"
            self.error_message = ""

        print("Stream stopped")

    def get_frame(self) -> Optional[bytes]:
        with self.frame_lock:
            return self.current_frame

    def capture_snapshot(self) -> Optional[Path]:
        with self.frame_lock:
            if not self.current_frame:
                return None
            data = self.current_frame

        path = self.snapshot_dir / f"snapshot_{int(time.time()*1000)}.jpg"
        with open(path, 'wb') as f:
            f.write(data)
        return path

    def set_zoom(self, percent: int) -> bool:
        if not self.session and not self._init_session():
            print(f"Zoom failed: no session")
            return False
        try:
            percent = max(0, min(100, percent))
            print(f"Setting zoom to {percent}%...")

            # Try webcam zoom endpoint first (some firmware versions)
            try:
                result = self._gopro_request(f"/gopro/webcam/zoom?percent={percent}", timeout=3)
                print(f"Webcam zoom result: {result}")
                return True
            except Exception as e1:
                print(f"Webcam zoom not available: {e1}")

            # Try standard digital zoom (may need to briefly exit webcam mode)
            if self.streaming:
                # Stop webcam, set zoom, restart
                print("Stopping webcam to change zoom...")
                try:
                    self._gopro_request("/gopro/webcam/stop", timeout=3)
                except:
                    pass
                time.sleep(0.3)

                result = self._gopro_request(f"/gopro/camera/digital_zoom?percent={percent}")
                print(f"Digital zoom result: {result}")

                # Restart webcam
                print("Restarting webcam...")
                self._gopro_request("/gopro/webcam/start", timeout=10)
                return True
            else:
                result = self._gopro_request(f"/gopro/camera/digital_zoom?percent={percent}")
                print(f"Digital zoom result: {result}")
                return True

        except Exception as e:
            self.error_message = str(e)
            print(f"Zoom error: {e}")
            return False

    def set_setting(self, setting_id: int, option_id: int) -> bool:
        """Set a GoPro camera setting."""
        if not self.session and not self._init_session():
            print(f"Set setting failed: no session")
            return False
        try:
            print(f"Setting {setting_id} to option {option_id}...")

            # If streaming, need to stop webcam first
            was_streaming = self.streaming
            if was_streaming:
                print("Pausing webcam to change setting...")
                try:
                    self._gopro_request("/gopro/webcam/stop", timeout=3)
                except:
                    pass
                time.sleep(0.3)

            result = self._gopro_request(f"/gopro/camera/setting?setting={setting_id}&option={option_id}")
            print(f"Setting result: {result}")

            # Restart webcam if it was running
            if was_streaming:
                print("Restarting webcam...")
                time.sleep(0.3)
                self._gopro_request("/gopro/webcam/start", timeout=10)

            return True
        except Exception as e:
            self.error_message = str(e)
            print(f"Setting error: {e}")
            # Try to restart webcam if we stopped it
            if self.streaming:
                try:
                    self._gopro_request("/gopro/webcam/start", timeout=10)
                except:
                    pass
            return False

    def get_status(self) -> dict:
        return {
            "status": self.status,
            "streaming": self.streaming,
            "frame_count": self.frame_count,
            "error": self.error_message or None
        }
