import os
import time
import uuid
import json
import shutil
import threading
import asyncio
import logging
import queue
from io import BytesIO
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, Response
from rembg import remove, new_session
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter
import requests

# Force ONNX Runtime to use CPU only (suppress TensorRT/CUDA warnings)
os.environ["ORT_TENSORRT_UNAVAILABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import onnxruntime
onnxruntime.set_default_logger_severity(3)  # ERROR only

from stream_manager import StreamManager

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(__file__), 'outputs')
app.config['GOPRO_PHOTOS'] = os.path.join(os.path.dirname(__file__), 'gopro_photos')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['GOPRO_PHOTOS'], exist_ok=True)

# rembg sessions (lazy loaded on first use)
rembg_sessions = {}

def get_rembg_session(model_name):
    """Get or create a rembg session for the given model (CPU only)."""
    if model_name not in rembg_sessions:
        print(f"Loading rembg model: {model_name}")
        rembg_sessions[model_name] = new_session(model_name, providers=["CPUExecutionProvider"])
    return rembg_sessions[model_name]

import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from mobile_sam import sam_model_registry, SamPredictor

sam_predictor = None
_models_loaded = False

def _load_models():
    """Load AI models. Called once in the worker process."""
    global sam_predictor, _models_loaded
    if _models_loaded:
        return
    _models_loaded = True

    print("Loading background removal models...")
    rembg_sessions['isnet-general-use'] = new_session("isnet-general-use", providers=["CPUExecutionProvider"])
    print("Default model loaded successfully!")

    print("Loading MobileSAM for interactive segmentation...")
    MOBILESAM_CHECKPOINT = os.path.join(os.path.dirname(__file__), 'weights', 'mobile_sam.pt')
    mobilesam_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mobile_sam = sam_model_registry['vit_t'](checkpoint=MOBILESAM_CHECKPOINT)
    mobile_sam.to(device=mobilesam_device)
    mobile_sam.eval()
    sam_predictor = SamPredictor(mobile_sam)

# In debug mode, Flask's reloader runs module-level code twice (parent + child).
# Only load in the child worker here; __main__ handles the initial run.
if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    _load_models()

# Cache for current image embedding (avoid re-encoding same image)
sam_image_cache = {
    'image_id': None,
    'predictor_ready': False
}


# ============================================================================
# Color Segment Selection (Click to Select) - Legacy, kept for compatibility
# ============================================================================

def get_color_segment(transparent_img, white_bg_img, click_x, click_y, tolerance=15):
    """
    Get a mask of the connected region around the click point with similar color.

    Uses flood-fill approach: only selects contiguous pixels near the click,
    not every pixel in the image that matches the color.

    Args:
        transparent_img: The RGBA image with transparency (what we'll modify)
        white_bg_img: The RGB image composited on white (what user clicks on)
        click_x, click_y: Click coordinates
        tolerance: Color difference tolerance for flood fill

    Returns: mask (numpy array), contour points for outline
    """
    import cv2

    # Ensure correct modes
    if transparent_img.mode != 'RGBA':
        transparent_img = transparent_img.convert('RGBA')
    if white_bg_img.mode != 'RGB':
        white_bg_img = white_bg_img.convert('RGB')

    trans_array = np.array(transparent_img)
    white_array = np.array(white_bg_img).copy()  # Copy because floodFill modifies it
    h, w = white_array.shape[:2]

    # Clamp click coordinates
    click_x = max(0, min(w - 1, int(click_x)))
    click_y = max(0, min(h - 1, int(click_y)))

    # Get the color at click point
    target_color = white_array[click_y, click_x, :3]
    print(f"Clicked at ({click_x}, {click_y}), color RGB: {target_color}")

    # Use OpenCV floodFill to select only the connected region
    # This is like the magic wand tool - only selects contiguous similar pixels

    # Create a mask for floodFill (needs to be h+2 x w+2)
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # floodFill parameters
    # loDiff and upDiff control how different neighboring pixels can be
    lo_diff = (tolerance, tolerance, tolerance)
    up_diff = (tolerance, tolerance, tolerance)

    # Perform flood fill from the click point
    # flags: 4 = 4-connectivity, 8 = 8-connectivity
    # Using 4-connectivity for more precise selection
    flags = 4 | (255 << 8)  # Fill with value 255, 4-connected

    cv2.floodFill(
        white_array,
        flood_mask,
        (click_x, click_y),
        (255, 0, 255),  # New color (doesn't matter, we use the mask)
        lo_diff,
        up_diff,
        flags
    )

    # Extract the actual mask (remove the 1-pixel border)
    final_mask = flood_mask[1:-1, 1:-1]

    # Also ensure we only affect pixels that have some opacity in transparent image
    alpha_mask = (trans_array[:, :, 3] > 10).astype(np.uint8)
    final_mask = cv2.bitwise_and(final_mask, alpha_mask * 255)

    # Find contours for the outline
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert contours to list of points for JSON
    contour_points = []
    for contour in contours:
        # Simplify contours to reduce data size
        epsilon = 2
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) > 2:
            points = approx.reshape(-1, 2).tolist()
            contour_points.append(points)

    pixel_count = np.sum(final_mask > 0)
    print(f"Flood-fill selected {pixel_count} pixels with tolerance {tolerance}, {len(contour_points)} contours")

    return final_mask, contour_points


def remove_color_segment(img, mask, dilate_pixels=None, feather_pixels=None):
    """
    Remove pixels indicated by mask (set alpha to 0).
    Dilates the mask slightly to catch shadow/edge remnants.
    Feathers the edge for smooth transition with drop shadow.

    Args:
        img: PIL Image to modify
        mask: Binary mask (255 = selected area to remove)
        dilate_pixels: How many pixels to expand the mask (None = auto from mask size)
        feather_pixels: How many pixels to feather the edge (None = auto from mask size)
    """
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # Auto-compute dilation and feathering from mask area ratio
    if dilate_pixels is None or feather_pixels is None:
        mask_area = np.sum(mask > 0)
        image_area = mask.shape[0] * mask.shape[1]
        ratio = mask_area / image_area if image_area > 0 else 0

        if dilate_pixels is None:
            if ratio < 0.05:
                dilate_pixels = 2   # Small selection: minimal eat-in
            elif ratio <= 0.25:
                dilate_pixels = 3   # Medium: balanced
            else:
                dilate_pixels = 4   # Large: more dilation for shadow catch

        if feather_pixels is None:
            if ratio < 0.05:
                feather_pixels = 4  # Small: generous smoothing
            elif ratio <= 0.25:
                feather_pixels = 3  # Medium: balanced
            else:
                feather_pixels = 2  # Large: less feather

    img_array = np.array(img)

    # Dilate the mask to catch shadow/edge remnants around the selection
    if dilate_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_pixels * 2 + 1, dilate_pixels * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)

    # Feather the mask edge for smooth alpha transition
    if feather_pixels > 0:
        # Create a soft falloff at the mask edge using Gaussian blur
        mask_float = mask.astype(np.float32) / 255.0
        mask_blurred = cv2.GaussianBlur(mask_float, (0, 0), sigmaX=feather_pixels)
        # The blurred mask creates a gradient at edges (0.0 to 1.0)
        # We want: inside mask = 0 alpha, outside = keep original, edge = gradient
        # So we subtract the blurred mask from the current alpha
        current_alpha = img_array[:, :, 3].astype(np.float32)
        new_alpha = current_alpha * (1.0 - mask_blurred)
        img_array[:, :, 3] = np.clip(new_alpha, 0, 255).astype(np.uint8)
    else:
        # Hard edge removal
        img_array[:, :, 3] = np.where(mask > 0, 0, img_array[:, :, 3])

    return Image.fromarray(img_array, 'RGBA')

# ==================== GoPro COHN Integration ====================

# GoPro credential files (created by provisioning)
GOPRO_CREDS_FILE = Path(__file__).parent / "cohn_credentials.json"
GOPRO_CERT_FILE = Path(__file__).parent / "cohn.crt"

# GoPro state
gopro_session = None
gopro_config = None
gopro_lock = threading.Lock()
gopro_photo_in_progress = False
gopro_photo_mode_set = False  # Track if we've set photo mode this session
gopro_last_preview = None  # Track last preview: {'path': Path, 'time': timestamp}

# Stream manager for real-time preview
stream_manager = StreamManager(GOPRO_CREDS_FILE, GOPRO_CERT_FILE, Path(app.config['GOPRO_PHOTOS']))

# GoPro COHN provisioning support
from gopro_cohn import provision_gopro

# Provisioning state (shared between routes and background thread)
provision_status = {
    'step': None,
    'message': '',
    'done': False,
    'error': None
}
provision_lock = threading.Lock()


def init_gopro_session():
    """Initialize GoPro HTTPS session with credentials."""
    global gopro_session, gopro_config

    if not GOPRO_CREDS_FILE.exists() or not GOPRO_CERT_FILE.exists():
        return False, "GoPro credentials not found. Run provisioning first."

    try:
        with open(GOPRO_CREDS_FILE) as f:
            gopro_config = json.load(f)

        gopro_session = requests.Session()
        gopro_session.verify = str(GOPRO_CERT_FILE)
        gopro_session.auth = (gopro_config["username"], gopro_config["password"])

        # Configure connection pooling
        adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1)
        gopro_session.mount("https://", adapter)

        return True, "GoPro session initialized"
    except Exception as e:
        return False, str(e)


def gopro_request(endpoint: str, timeout: int = 10, retries: int = 3) -> dict:
    """Make a request to the GoPro API with retry logic."""
    global gopro_session, gopro_config

    if gopro_session is None:
        success, msg = init_gopro_session()
        if not success:
            raise Exception(msg)

    url = f"https://{gopro_config['ip_address']}{endpoint}"

    last_error = None
    for attempt in range(retries):
        try:
            with gopro_lock:
                response = gopro_session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.HTTPError as e:
            last_error = e
            if e.response.status_code == 503:
                time.sleep(1)
            else:
                raise
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_error = e
            time.sleep(2)

    raise last_error


def gopro_download_media(file_path: str) -> Path:
    """Download a media file from the GoPro."""
    global gopro_session, gopro_config

    url = f"https://{gopro_config['ip_address']}/videos/DCIM/{file_path}"

    with gopro_lock:
        response = gopro_session.get(url, timeout=60)
    response.raise_for_status()

    filename = file_path.split("/")[-1]
    local_path = Path(app.config['GOPRO_PHOTOS']) / filename

    with open(local_path, "wb") as f:
        f.write(response.content)

    return local_path


def gopro_ensure_photo_mode_once():
    """Set photo mode once per session. Returns True if ready."""
    global gopro_photo_mode_set

    if gopro_photo_mode_set:
        return True  # Already set this session

    try:
        print("Setting GoPro to photo mode (once per session)...")
        gopro_request("/gopro/camera/presets/set_group?id=1001", timeout=5)
        time.sleep(0.3)
        gopro_photo_mode_set = True
        print("Photo mode set successfully")
        return True
    except Exception as e:
        print(f"Warning: Could not set photo mode: {e}")
        # Still allow capture - user may have set it manually
        gopro_photo_mode_set = True  # Don't keep retrying
        return True


def gopro_get_latest_photo():
    """Get the latest photo from camera."""
    media = gopro_request("/gopro/media/list", retries=5)

    latest_photo = None
    latest_folder = None

    for folder_info in media.get("media", []):
        folder = folder_info.get("d", "")
        files = folder_info.get("fs", [])
        for file_info in files:
            filename = file_info.get("n", "")
            if filename.lower().endswith((".jpg", ".jpeg")):
                latest_photo = filename
                latest_folder = folder

    return latest_folder, latest_photo

MAX_SIZE = 2048  # Higher resolution for better quality

# Default model for background removal
DEFAULT_MODEL = 'u2netp'


def remove_background_ai(img, model='isnet'):
    """
    AI-based background removal using rembg models.

    Models available:
    - isnet: ISNet general use - best quality, ~1.5s (recommended)
    - u2netp: Fast U2Net - fast, ~0.3s
    - u2net: Full U2Net - high quality, ~0.5s
    - silueta: Fastest model - very fast, ~0.2s

    Use manual selection tools for post-processing cleanup if needed.
    """

    # Map model names to rembg session names
    rembg_model_map = {
        'u2netp': 'u2netp',
        'silueta': 'silueta',
        'isnet': 'isnet-general-use',
        'u2net': 'u2net',
    }

    if model in rembg_model_map:
        session = get_rembg_session(rembg_model_map[model])
    else:
        session = get_rembg_session('isnet-general-use')

    # AI background removal
    result = remove(img, session=session)
    arr = np.array(result)

    import cv2

    alpha = arr[:, :, 3]

    # Zero out noise
    alpha = np.where(alpha < 30, 0, alpha).astype(np.uint8)

    # Edge decontamination: the AI model leaves a 2-3px border of
    # semi-transparent or fully-opaque pixels whose RGB is contaminated
    # with background color (visible as a grey outline on white BGs).
    # 1) Erode alpha by 3px to cut into the contaminated edge zone
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    alpha = cv2.erode(alpha, kernel, iterations=1)
    # 2) Harden: push low-alpha edge remnants to zero
    alpha = np.where(alpha < 100, 0, alpha).astype(np.uint8)
    # 3) Re-soften the new clean edge so it's not jaggy
    alpha = cv2.GaussianBlur(alpha, (3, 3), sigmaX=0.7)

    arr[:, :, 3] = alpha

    return Image.fromarray(arr, 'RGBA')


# ============================================================================
# Fast Color Key Background Removal
# ============================================================================

def remove_background_colorkey(img, key_color='green', tolerance=40, spill_removal=True, edge_feather=2):
    """
    Fast chroma key (green/blue screen) background removal.
    Handles shadows on green screen by using multiple detection passes.
    ~0.05-0.15 seconds on CPU.

    key_color: 'green', 'white', or RGB tuple like (0, 255, 0)
    tolerance: How much variation from key color to remove (0-100)
    spill_removal: Remove color spill on edges
    edge_feather: Pixels of edge feathering for smooth edges (0-5)
    """
    import cv2

    img_array = np.array(img.convert('RGB'))
    h, w = img_array.shape[:2]

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    if key_color == 'green':
        # Multi-pass approach to handle shadows on green screen
        # Pass 1: Bright green (well-lit areas)
        lower1 = np.array([35, 80, 80])
        upper1 = np.array([85, 255, 255])

        # Pass 2: Shadowed green (darker, less saturated)
        lower2 = np.array([30, 30, 30])
        upper2 = np.array([90, 255, 180])

        # Pass 3: Very dark shadows that are still greenish
        lower3 = np.array([25, 15, 15])
        upper3 = np.array([95, 200, 120])

        # Adjust ranges based on tolerance
        tolerance_factor = tolerance / 40  # Scale tolerance

        # Expand hue range
        hue_expand = int(15 * tolerance_factor)
        lower1[0] = max(0, lower1[0] - hue_expand)
        upper1[0] = min(180, upper1[0] + hue_expand)
        lower2[0] = max(0, lower2[0] - hue_expand)
        upper2[0] = min(180, upper2[0] + hue_expand)
        lower3[0] = max(0, lower3[0] - hue_expand)
        upper3[0] = min(180, upper3[0] + hue_expand)

        # Lower saturation threshold for shadows
        sat_reduce = int(30 * tolerance_factor)
        lower1[1] = max(0, lower1[1] - sat_reduce)
        lower2[1] = max(0, lower2[1] - sat_reduce)
        lower3[1] = max(0, lower3[1] - sat_reduce)

        # Lower value threshold for dark shadows
        val_reduce = int(40 * tolerance_factor)
        lower1[2] = max(0, lower1[2] - val_reduce)
        lower2[2] = max(0, lower2[2] - val_reduce)
        lower3[2] = max(0, lower3[2] - val_reduce)

        # Create masks for each pass
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask3 = cv2.inRange(hsv, lower3, upper3)

        # Combine all green detection masks
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.bitwise_or(mask, mask3)

    elif key_color == 'white':
        # White/light gray background - also handle shadows
        lower1 = np.array([0, 0, 180])
        upper1 = np.array([180, 40, 255])

        # Shadowed white/gray
        lower2 = np.array([0, 0, 120])
        upper2 = np.array([180, 30, 200])

        tolerance_factor = tolerance / 40
        val_reduce = int(60 * tolerance_factor)
        lower1[2] = max(0, lower1[2] - val_reduce)
        lower2[2] = max(0, lower2[2] - val_reduce)

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

    elif isinstance(key_color, tuple):
        color_hsv = cv2.cvtColor(np.uint8([[list(key_color)]]), cv2.COLOR_RGB2HSV)[0][0]
        lower = np.array([max(0, color_hsv[0] - 15), 30, 30])
        upper = np.array([min(180, color_hsv[0] + 15), 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    else:
        lower = np.array([35, 50, 50])
        upper = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

    # Invert mask (we want foreground, not background)
    mask = cv2.bitwise_not(mask)

    # Light morphological cleanup - just remove tiny noise specks
    kernel_small = np.ones((3, 3), np.uint8)

    # Single pass to fill tiny holes (less aggressive)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    # Single pass to remove tiny noise specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Always apply minimal anti-aliasing for smooth edges (even when feather=0)
    mask_float = mask.astype(np.float32) / 255.0

    # Base smoothing: very subtle Gaussian blur for anti-aliasing
    base_blur = 3  # Minimal blur for clean edges
    mask_float = cv2.GaussianBlur(mask_float, (base_blur, base_blur), 0)

    # Additional feathering if requested
    if edge_feather > 0:
        blur_amount = edge_feather * 2 + 1  # Must be odd (3, 5, 7, 9, 11)
        mask_float = cv2.GaussianBlur(mask_float, (blur_amount, blur_amount), 0)

    # Restore contrast to keep edges defined but smooth
    # This prevents the mask from becoming too soft/transparent
    mask_float = np.clip((mask_float - 0.2) * 1.5, 0, 1)
    mask = (mask_float * 255).astype(np.uint8)

    # Remove green spill from edges
    if spill_removal and key_color == 'green':
        img_array = remove_green_spill(img_array, mask)

    # Create RGBA result
    result = np.dstack((img_array, mask))

    return Image.fromarray(result, 'RGBA')


def remove_green_spill(img_array, mask):
    """Remove green color spill from edges of the subject."""
    import cv2

    # Find edge pixels (where mask transitions)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=2)
    eroded = cv2.erode(mask, kernel, iterations=2)
    edge_mask = dilated - eroded

    # Desaturate green in edge areas
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Find greenish pixels in edge area
    green_mask = (hsv[:,:,0] > 35) & (hsv[:,:,0] < 85) & (edge_mask > 0)

    # Reduce saturation of green pixels
    hsv[green_mask, 1] = hsv[green_mask, 1] * 0.3

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def resize_image(img, max_size=MAX_SIZE):
    """Resize image maintaining aspect ratio if larger than max_size."""
    width, height = img.size
    if width <= max_size and height <= max_size:
        return img, False

    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))

    return img.resize((new_width, new_height), Image.LANCZOS), True


def apply_alpha_threshold(img, threshold):
    """Apply threshold to alpha channel to adjust edge hardness."""
    if img.mode != 'RGBA':
        return img

    data = np.array(img)
    alpha = data[:, :, 3].astype(float)

    # Threshold is 0-100, convert to 0-255 range for processing
    # Lower threshold = softer edges, higher = harder edges
    thresh_value = threshold * 2.55

    # Apply threshold with smooth transition
    if threshold < 50:
        # Softer edges - expand the semi-transparent region
        factor = (50 - threshold) / 50
        alpha = np.where(alpha > 0, np.clip(alpha + (255 - alpha) * factor * 0.3, 0, 255), alpha)
    else:
        # Harder edges - make semi-transparent pixels more opaque or transparent
        factor = (threshold - 50) / 50
        alpha = np.where(alpha > 128, np.clip(alpha + (255 - alpha) * factor, 0, 255),
                        np.clip(alpha * (1 - factor), 0, 255))

    data[:, :, 3] = alpha.astype(np.uint8)
    return Image.fromarray(data)


def save_undo_backup(image_id):
    """Save current adjusted (or output) image as undo backup before editing."""
    adjusted_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_adjusted.png')
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')
    undo_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_undo.png')
    src = adjusted_path if os.path.exists(adjusted_path) else output_path
    if os.path.exists(src):
        shutil.copy2(src, undo_path)


def apply_mask_corrections(img, mask_data, original_size, image_id=None):
    """Apply manual mask corrections from the canvas."""
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    data = np.array(img)

    # mask_data contains {keep: [[x,y],...], remove: [[x,y],...], brushSize: int}
    keep_points = mask_data.get('keep', [])
    remove_points = mask_data.get('remove', [])
    brush_size = mask_data.get('brushSize', 10)

    # Scale points from display size to actual image size
    img_width, img_height = img.size
    orig_width, orig_height = original_size
    scale_x = img_width / orig_width if orig_width else 1
    scale_y = img_height / orig_height if orig_height else 1

    # Create coordinate grids for distance calculation
    y_coords, x_coords = np.ogrid[:img_height, :img_width]

    # Load original image for "keep" restoration
    original_data = None
    if keep_points and image_id:
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{image_id}_original.png')
        if os.path.exists(original_path):
            original_img = Image.open(original_path).convert('RGBA')
            original_data = np.array(original_img)

    # Apply keep points (restore original pixels with full alpha)
    for point in keep_points:
        px, py = int(point[0] * scale_x), int(point[1] * scale_y)
        mask = (x_coords - px) ** 2 + (y_coords - py) ** 2 <= brush_size ** 2
        if original_data is not None:
            # Restore RGB from original and set alpha to 255
            for c in range(3):
                data[:, :, c] = np.where(mask, original_data[:, :, c], data[:, :, c])
        data[:, :, 3] = np.where(mask, 255, data[:, :, 3])

    # Apply remove points (set alpha to 0)
    for point in remove_points:
        px, py = int(point[0] * scale_x), int(point[1] * scale_y)
        mask = (x_coords - px) ** 2 + (y_coords - py) ** 2 <= brush_size ** 2
        data[:, :, 3] = np.where(mask, 0, data[:, :, 3])

    return Image.fromarray(data)


def detect_fragments(img, alpha_threshold=10, min_size=50):
    """Detect connected fragments using alpha channel."""
    if img.mode != 'RGBA':
        return None, {}

    data = np.array(img)
    alpha = data[:, :, 3]

    # Binary mask from alpha > threshold
    binary_mask = alpha > alpha_threshold

    # Label connected components (8-connectivity)
    labeled = label(binary_mask, connectivity=2)

    # Get properties of each region
    regions = regionprops(labeled)

    # Find the largest fragment (main subject)
    max_area = 0
    main_label = 0
    for region in regions:
        if region.area > max_area:
            max_area = region.area
            main_label = region.label

    # Build fragment info
    fragment_info = {}
    for region in regions:
        if region.area < min_size:
            continue

        classification = classify_fragment(region.area, max_area)

        fragment_info[int(region.label)] = {
            'area': int(region.area),
            'centroid': [float(region.centroid[1]), float(region.centroid[0])],  # x, y
            'bounds': [int(region.bbox[1]), int(region.bbox[0]),
                      int(region.bbox[3]), int(region.bbox[2])],  # min_x, min_y, max_x, max_y
            'classification': classification,
            'is_main': region.label == main_label
        }

    return labeled, fragment_info


def classify_fragment(area, max_area):
    """Classify fragment as 'main', 'secondary', 'artifact', or 'tiny'."""
    ratio = area / max_area if max_area > 0 else 0

    if ratio > 0.5:
        return 'main'
    elif ratio > 0.1:
        return 'secondary'
    elif ratio > 0.01:
        return 'artifact'
    else:
        return 'tiny'


def create_fragment_overlay(labeled, fragment_info, selected_remove):
    """Create overlay highlighting fragments (yellow for artifacts, red for selected)."""
    h, w = labeled.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    for frag_id, info in fragment_info.items():
        mask = labeled == frag_id

        if frag_id in selected_remove:
            # Red for selected to remove
            overlay[mask] = [255, 50, 50, 180]
        elif info['is_main']:
            # Very subtle green for main subject
            overlay[mask] = [50, 255, 100, 40]
        elif info['classification'] in ['artifact', 'tiny']:
            # Yellow for artifacts
            overlay[mask] = [255, 200, 50, 100]
        else:
            # Light blue for secondary
            overlay[mask] = [100, 200, 255, 80]

    return Image.fromarray(overlay, 'RGBA')


def remove_fragments(img, labeled, fragment_ids):
    """Remove specified fragments by replacing with white background."""
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    data = np.array(img)

    for frag_id in fragment_ids:
        mask = labeled == frag_id
        # Set to white with full opacity
        data[mask] = [255, 255, 255, 255]

    return Image.fromarray(data)


def soften_edges(img, radius=1.5):
    """Apply gentle edge softening by blurring the alpha channel."""
    if img.mode != 'RGBA':
        return img

    data = np.array(img).astype(float)
    alpha = data[:, :, 3]

    # Apply gaussian blur to alpha channel for soft edges
    alpha_soft = gaussian_filter(alpha, sigma=radius)

    data[:, :, 3] = alpha_soft
    return Image.fromarray(data.astype(np.uint8))


def decontaminate_edges(img, edge_threshold=200):
    """
    Remove color contamination from edge pixels.
    Edge pixels often contain background color blended in.
    This replaces edge pixel colors with colors from nearby interior pixels.
    """
    from scipy.ndimage import binary_dilation, distance_transform_edt

    if img.mode != 'RGBA':
        return img

    data = np.array(img)
    alpha = data[:, :, 3]

    # Define regions
    # Interior: fully opaque pixels (alpha >= edge_threshold)
    # Edge: semi-transparent pixels (0 < alpha < edge_threshold)
    interior_mask = alpha >= edge_threshold
    edge_mask = (alpha > 0) & (alpha < edge_threshold)

    if not np.any(edge_mask):
        return img  # No edge pixels to fix

    # For each edge pixel, find the nearest interior pixel and use its color
    # Use distance transform to find nearest interior pixel indices

    # Create a copy for the result
    result = data.copy()

    # Get indices of interior pixels
    if not np.any(interior_mask):
        return img  # No interior pixels to sample from

    # Use distance transform with indices to find nearest interior pixel for each location
    # distance_transform_edt returns distances and can give indices
    dist, indices = distance_transform_edt(~interior_mask, return_indices=True)

    # For edge pixels, replace RGB with the color of the nearest interior pixel
    edge_y, edge_x = np.where(edge_mask)
    nearest_y = indices[0][edge_y, edge_x]
    nearest_x = indices[1][edge_y, edge_x]

    # Copy RGB from nearest interior pixel (keep original alpha)
    result[edge_y, edge_x, 0] = data[nearest_y, nearest_x, 0]
    result[edge_y, edge_x, 1] = data[nearest_y, nearest_x, 1]
    result[edge_y, edge_x, 2] = data[nearest_y, nearest_x, 2]

    return Image.fromarray(result)


def clean_edges(img, erode_pixels=2, threshold=128):
    """
    Clean up edges by:
    1. Eroding the mask slightly to remove contaminated edge pixels
    2. Thresholding alpha to remove semi-transparent pixels
    """
    from scipy.ndimage import binary_erosion

    if img.mode != 'RGBA':
        return img

    data = np.array(img)
    alpha = data[:, :, 3]

    # Create binary mask and erode it
    mask = alpha > threshold

    # Erode to remove outer contaminated pixels
    if erode_pixels > 0:
        struct = np.ones((erode_pixels * 2 + 1, erode_pixels * 2 + 1))
        mask = binary_erosion(mask, structure=struct)

    # Apply the eroded mask - pixels outside are fully transparent
    data[:, :, 3] = np.where(mask, 255, 0).astype(np.uint8)

    return Image.fromarray(data)


def add_drop_shadow(img, offset=(-8, 12), shadow_blur=15, shadow_opacity=0.35):
    """Add a drop shadow to the image. Light source from top-right."""
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    data = np.array(img)
    alpha = data[:, :, 3].astype(float) / 255.0

    # Create shadow from alpha channel
    shadow = gaussian_filter(alpha, sigma=shadow_blur)

    # Offset the shadow (negative x = left, positive y = down)
    offset_x, offset_y = offset
    shadow_shifted = np.zeros_like(shadow)

    # Calculate source and destination slices for the shift
    src_y_start = max(0, -offset_y)
    src_y_end = shadow.shape[0] - max(0, offset_y)
    src_x_start = max(0, -offset_x)
    src_x_end = shadow.shape[1] - max(0, offset_x)

    dst_y_start = max(0, offset_y)
    dst_y_end = shadow.shape[0] + min(0, offset_y)
    dst_x_start = max(0, offset_x)
    dst_x_end = shadow.shape[1] + min(0, offset_x)

    shadow_shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        shadow[src_y_start:src_y_end, src_x_start:src_x_end]

    shadow_shifted = (shadow_shifted * shadow_opacity * 255).astype(np.uint8)

    return img, shadow_shifted


def load_watermark():
    """Load the watermark logo."""
    logo_path = os.path.join(os.path.dirname(__file__), 'static', 'logo.png')
    if os.path.exists(logo_path):
        return Image.open(logo_path).convert('RGBA')
    return None

# Cache the watermark
WATERMARK = None

def get_watermark():
    """Get cached watermark or load it."""
    global WATERMARK
    if WATERMARK is None:
        WATERMARK = load_watermark()
    return WATERMARK


def get_square_transform(img, target_size=2048, zoom=100, max_scale=None):
    """
    Compute the transformation parameters matching fit_subject_to_square.
    Returns dict with x_min, y_min, scale, paste_x, paste_y or None if no subject.
    max_scale: if set, caps upscaling to this factor (e.g. 2.0 = max 2x enlargement).
    """
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    data = np.array(img)
    h, w = data.shape[:2]
    alpha = data[:, :, 3]

    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)

    if not rows.any() or not cols.any():
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    margin = 2
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(w - 1, x_max + margin)
    y_max = min(h - 1, y_max + margin)

    subj_w = x_max + 1 - x_min
    subj_h = y_max + 1 - y_min

    base_fill = 0.92
    available = target_size * base_fill
    base_scale = min(available / subj_w, available / subj_h)
    if max_scale is not None:
        base_scale = min(base_scale, max_scale)
    scale = base_scale * (zoom / 100.0)

    new_w = max(1, int(subj_w * scale))
    new_h = max(1, int(subj_h * scale))

    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2

    return {
        'x_min': x_min, 'y_min': y_min,
        'scale': scale,
        'paste_x': paste_x, 'paste_y': paste_y
    }


def fit_subject_to_square(img, target_size=2048, zoom=100, max_scale=None):
    """
    Fit the subject (non-transparent area) to a square canvas.
    Crops to subject bounding box and centers on square canvas.
    Preserves native resolution at zoom=100 — no resampling so fabric
    detail, textures, and fine features stay pixel-sharp.

    zoom: 100 = fill 92% of canvas, >100 = larger, <100 = smaller
    max_scale: if set, caps upscaling to this factor (e.g. 2.0 = max 2x enlargement).
    """
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    data = np.array(img)
    alpha = data[:, :, 3]

    # Find bounding box of non-transparent pixels
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)

    if not rows.any() or not cols.any():
        return Image.new('RGBA', (target_size, target_size), (0, 0, 0, 0))

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Crop to subject with small margin
    margin = 2
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(data.shape[1] - 1, x_max + margin)
    y_max = min(data.shape[0] - 1, y_max + margin)

    subject = img.crop((x_min, y_min, x_max + 1, y_max + 1))
    subj_w, subj_h = subject.size

    # Scale subject to fill the canvas with margin
    base_fill = 0.92  # 4% margin on each side
    available = target_size * base_fill
    base_scale = min(available / subj_w, available / subj_h)
    if max_scale is not None:
        base_scale = min(base_scale, max_scale)
    scale = base_scale * (zoom / 100.0)

    new_w = max(1, int(subj_w * scale))
    new_h = max(1, int(subj_h * scale))

    if scale != 1.0:
        subject_scaled = subject.resize((new_w, new_h), Image.LANCZOS)
    else:
        subject_scaled = subject
    subject_data = np.array(subject_scaled)

    # Center on target canvas
    canvas_data = np.zeros((target_size, target_size, 4), dtype=np.uint8)

    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2

    # Calculate visible region (handles subject larger than canvas)
    src_y1 = max(0, -paste_y)
    src_x1 = max(0, -paste_x)
    dst_y1 = max(0, paste_y)
    dst_x1 = max(0, paste_x)

    h = min(new_h - src_y1, target_size - dst_y1)
    w = min(new_w - src_x1, target_size - dst_x1)

    if h > 0 and w > 0:
        canvas_data[dst_y1:dst_y1+h, dst_x1:dst_x1+w] = subject_data[src_y1:src_y1+h, src_x1:src_x1+w]

    return Image.fromarray(canvas_data)


def pad_to_square(img, target_size=2048):
    """
    Pad image to a square canvas without cropping or scaling.
    Preserves original framing so the after image matches the before image
    composition — no subject appears 'zoomed' and coordinates map 1:1.
    """
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    w, h = img.size
    canvas_size = max(w, h, target_size)

    if w == canvas_size and h == canvas_size:
        return img

    canvas = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
    paste_x = (canvas_size - w) // 2
    paste_y = (canvas_size - h) // 2
    canvas.paste(img, (paste_x, paste_y), img)
    return canvas


def auto_brighten_for_white_bg(img_data):
    """Gently brighten subject via gamma correction for white-background display.
    Prevents the 'looks darker on white' perceptual effect common in e-commerce."""
    alpha = img_data[:, :, 3]
    mask = alpha > 128
    if not np.any(mask):
        return img_data

    rgb = img_data[:, :, :3].astype(np.float32)
    lum = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
    median_lum = float(np.median(lum[mask]))

    target = 155.0
    if median_lum >= target:
        return img_data

    gamma = np.log(target / 255.0) / np.log(max(median_lum, 1.0) / 255.0)
    gamma = float(np.clip(gamma, 0.75, 1.0))

    lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)]).astype(np.uint8)

    result = img_data.copy()
    for c in range(3):
        result[:, :, c] = lut[img_data[:, :, c]]
    return result


def add_watermark(img, watermark_scale=0.075, watermark_opacity=0.3, watermark_margin=15):
    """Add watermark to any RGB image. Returns RGB image with watermark composited."""
    w, h = img.size
    if img.mode != 'RGB':
        img = img.convert('RGB')

    watermark = get_watermark()
    if watermark is not None:
        # Scale watermark relative to image size
        wm_width = int(w * watermark_scale)
        wm_height = int(watermark.height * (wm_width / watermark.width))
        wm_resized = watermark.resize((wm_width, wm_height), Image.LANCZOS)

        # Adjust opacity
        wm_data = np.array(wm_resized)
        wm_data[:, :, 3] = (wm_data[:, :, 3] * watermark_opacity).astype(np.uint8)
        wm_resized = Image.fromarray(wm_data)

        # Position in bottom-left
        wm_x = watermark_margin
        wm_y = h - wm_height - watermark_margin

        # Composite watermark
        img = img.convert('RGBA')
        img.paste(wm_resized, (wm_x, wm_y), wm_resized)
        img = img.convert('RGB')

    return img


def prepare_ecommerce_image(img, watermark_scale=0.075, watermark_opacity=0.3, watermark_margin=15):
    """Prepare image for e-commerce: white background, watermark."""

    img_data = np.array(img)
    h, w = img_data.shape[:2]

    # Auto-brighten subject for white background
    img_data = auto_brighten_for_white_bg(img_data)

    # Start with white background
    result = np.full((h, w, 3), 255, dtype=np.uint8)

    # Composite on white - SAM refinement already cleaned up the edges
    alpha = img_data[:, :, 3].astype(float) / 255.0
    for c in range(3):
        result[:, :, c] = (img_data[:, :, c] * alpha + result[:, :, c] * (1 - alpha)).astype(np.uint8)

    result_img = Image.fromarray(result, 'RGB')

    return add_watermark(result_img, watermark_scale, watermark_opacity, watermark_margin)


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    timings = {}
    total_start = time.time()

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400


    try:
        # Generate unique ID for this image
        image_id = str(uuid.uuid4())

        # Read and process image
        step_start = time.time()
        img = Image.open(file.stream)
        if img.mode == 'RGBA':
            # Convert RGBA to RGB with white background for processing
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        timings['read'] = round((time.time() - step_start) * 1000)

        # Skip fisheye correction for uploads - uploaded images from phones/DSLRs
        # typically don't have barrel distortion (unlike GoPro/webcam captures)
        timings['lens_correct'] = 0

        # Store original size for mask corrections
        original_size = img.size

        # Resize if needed
        step_start = time.time()
        img, was_resized = resize_image(img)
        timings['resize'] = round((time.time() - step_start) * 1000)

        # Save original for comparison
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{image_id}_original.png')
        img.save(original_path, 'PNG')

        # Remove background using hybrid method (AI + green cleanup)
        step_start = time.time()
        model = request.form.get('model', 'u2netp')
        output = remove_background_ai(img, model=model)
        timings['method'] = model
        timings['remove_bg'] = round((time.time() - step_start) * 1000)

        # Skip edge erosion to preserve thin details like straps
        timings['clean_edges'] = 0

        # Save processed image
        step_start = time.time()
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')
        output.save(output_path, 'PNG')
        timings['save'] = round((time.time() - step_start) * 1000)

        timings['total'] = round((time.time() - total_start) * 1000)

        return jsonify({
            'success': True,
            'image_id': image_id,
            'timings': timings,
            'original_size': list(original_size),
            'processed_size': list(img.size),
            'was_resized': was_resized
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/use_original', methods=['POST'])
def use_original():
    """Skip background removal — use the original photo as the output."""
    data = request.json
    image_id = data.get('image_id') if data else None

    if not image_id:
        return jsonify({'error': 'No image ID provided'}), 400

    original_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{image_id}_original.png')
    if not os.path.exists(original_path):
        return jsonify({'error': 'Original image not found'}), 404

    try:
        img = Image.open(original_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')
        img.save(output_path, 'PNG')

        return jsonify({
            'success': True,
            'image_id': image_id,
            'use_original': True,
            'timings': {'remove_bg': 0, 'method': 'original'},
            'original_size': list(img.size),
            'processed_size': list(img.size),
            'was_resized': False
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/adjust', methods=['POST'])
def adjust():
    """Adjust the processed image with threshold, mask corrections, or zoom."""
    data = request.json
    image_id = data.get('image_id')
    threshold = data.get('threshold', 50)
    mask_data = data.get('mask_data')
    original_size = data.get('original_size', [0, 0])
    subject_zoom = data.get('subject_zoom', 100)  # 100-400%
    display_zoom = data.get('zoom', None)  # zoom level from fit_subject_to_square

    if not image_id:
        return jsonify({'error': 'No image ID provided'}), 400

    try:
        save_undo_backup(image_id)

        # Load from adjusted if exists, otherwise from output (cumulative edits)
        adjusted_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_adjusted.png')
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')

        if os.path.exists(adjusted_path):
            img = Image.open(adjusted_path)
        elif os.path.exists(output_path):
            img = Image.open(output_path)
        else:
            return jsonify({'error': 'Image not found'}), 404

        # If zoom is provided, erase points are in 2048 square space.
        # Transform them to original image space before applying.
        if display_zoom is not None and mask_data and (mask_data.get('keep') or mask_data.get('remove')):
            sq_transform = get_square_transform(img, 2048, display_zoom, max_scale=2.0)
            if sq_transform:
                for key in ('keep', 'remove'):
                    points = mask_data.get(key, [])
                    if points:
                        mask_data[key] = [
                            [
                                (p[0] - sq_transform['paste_x']) / sq_transform['scale'] + sq_transform['x_min'],
                                (p[1] - sq_transform['paste_y']) / sq_transform['scale'] + sq_transform['y_min']
                            ]
                            for p in points
                        ]
                # Also scale brush size from 2048 space to original space
                if mask_data.get('brushSize'):
                    mask_data['brushSize'] = int(mask_data['brushSize'] / sq_transform['scale'])
                # Set original_size to actual image size so apply_mask_corrections scales 1:1
                original_size = [img.size[0], img.size[1]]

        # Apply threshold adjustment
        if threshold != 50:
            img = apply_alpha_threshold(img, threshold)

        # Apply mask corrections if provided
        if mask_data and (mask_data.get('keep') or mask_data.get('remove')):
            img = apply_mask_corrections(img, mask_data, original_size, image_id)

        # Apply subject zoom if provided
        if subject_zoom > 100:
            img = apply_subject_zoom(img, subject_zoom)

        # Save adjusted image
        img.save(adjusted_path, 'PNG')

        return jsonify({
            'success': True,
            'image_id': image_id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/color_correct', methods=['POST'])
def color_correct():
    """Apply color temperature, tint, and brightness correction to the processed image."""
    data = request.json
    image_id = data.get('image_id')
    temperature = data.get('temperature', 0)  # -100 to +100
    tint = data.get('tint', 0)  # -100 to +100
    brightness = data.get('brightness', 0)  # -100 to +100

    if not image_id:
        return jsonify({'error': 'No image ID provided'}), 400

    try:
        save_undo_backup(image_id)

        # Load from adjusted if exists, otherwise from output
        adjusted_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_adjusted.png')
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')

        if os.path.exists(adjusted_path):
            img = Image.open(adjusted_path)
        elif os.path.exists(output_path):
            img = Image.open(output_path)
        else:
            return jsonify({'error': 'Image not found'}), 404

        # Apply color correction
        if temperature != 0 or tint != 0 or brightness != 0:
            img = apply_color_correction(img, temperature, tint, brightness)

        # Save adjusted image
        img.save(adjusted_path, 'PNG')

        return jsonify({
            'success': True,
            'image_id': image_id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/reset_colors', methods=['POST'])
def reset_colors():
    """Reset color corrections by removing the adjusted image."""
    data = request.json
    image_id = data.get('image_id')

    if not image_id:
        return jsonify({'error': 'No image ID provided'}), 400

    try:
        adjusted_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_adjusted.png')

        # Delete the adjusted file if it exists
        if os.path.exists(adjusted_path):
            os.remove(adjusted_path)

        return jsonify({
            'success': True,
            'image_id': image_id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/undo', methods=['POST'])
def undo():
    """Revert to the previous image state (one level of undo)."""
    data = request.json
    image_id = data.get('image_id')

    if not image_id:
        return jsonify({'error': 'No image ID provided'}), 400

    try:
        undo_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_undo.png')
        adjusted_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_adjusted.png')

        if not os.path.exists(undo_path):
            return jsonify({'error': 'Nothing to undo'}), 404

        shutil.copy2(undo_path, adjusted_path)
        os.remove(undo_path)

        # Invalidate SAM cache since image changed
        global sam_image_cache
        if sam_image_cache.get('image_id') == image_id:
            sam_image_cache['image_id'] = None
            sam_image_cache['predictor_ready'] = False

        return jsonify({
            'success': True,
            'image_id': image_id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/apply_effects', methods=['POST'])
def apply_effects():
    """Apply edge smoothing and/or drop shadow to the processed image."""
    data = request.json
    image_id = data.get('image_id')
    edge_smooth = data.get('edge_smooth', 0)  # 0-3 (0=none, 1=light, 2=medium, 3=heavy)
    shadow = data.get('shadow', 0)  # 0-3 (0=none, 1=soft, 2=medium, 3=strong)

    if not image_id:
        return jsonify({'error': 'No image ID provided'}), 400

    try:
        # Load from adjusted if exists, otherwise from output
        adjusted_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_adjusted.png')
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')

        if os.path.exists(adjusted_path):
            img = Image.open(adjusted_path)
        elif os.path.exists(output_path):
            img = Image.open(output_path)
        else:
            return jsonify({'error': 'Image not found'}), 404

        # Apply edge smoothing
        if edge_smooth > 0:
            img = apply_edge_smoothing(img, edge_smooth)

        # Apply drop shadow
        if shadow > 0:
            img = apply_drop_shadow(img, shadow)

        # Save adjusted image
        img.save(adjusted_path, 'PNG')

        return jsonify({
            'success': True,
            'image_id': image_id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/fragments', methods=['POST'])
def fragments():
    """Detect fragments in the processed image."""
    data = request.json
    image_id = data.get('image_id')
    alpha_threshold = data.get('alpha_threshold', 10)
    min_size = data.get('min_size', 50)

    if not image_id:
        return jsonify({'error': 'No image ID provided'}), 400

    try:
        # Load the processed image (adjusted if exists, otherwise output)
        adjusted_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_adjusted.png')
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')
        img_path = adjusted_path if os.path.exists(adjusted_path) else output_path

        if not os.path.exists(img_path):
            return jsonify({'error': 'Image not found'}), 404

        img = Image.open(img_path)

        # Detect fragments
        step_start = time.time()
        labeled, fragment_info = detect_fragments(img, alpha_threshold, min_size)
        detect_time = round((time.time() - step_start) * 1000)

        if labeled is None:
            return jsonify({'error': 'Image must have alpha channel'}), 400

        # Save labeled array for later use
        labeled_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_labeled.npy')
        np.save(labeled_path, labeled)

        # Save fragment info as JSON for later use
        info_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_fragment_info.json')
        with open(info_path, 'w') as f:
            json.dump(fragment_info, f)

        return jsonify({
            'success': True,
            'image_id': image_id,
            'fragment_count': len(fragment_info),
            'fragment_info': fragment_info,
            'detect_time': detect_time
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/fragments/click', methods=['POST'])
def fragments_click():
    """Find which fragment was clicked."""
    data = request.json
    image_id = data.get('image_id')
    x = data.get('x', 0)
    y = data.get('y', 0)

    if not image_id:
        return jsonify({'error': 'No image ID provided'}), 400

    try:
        # Load labeled array
        labeled_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_labeled.npy')
        if not os.path.exists(labeled_path):
            return jsonify({'error': 'Fragments not detected. Run fragment detection first.'}), 404

        labeled = np.load(labeled_path)

        # Load fragment info
        info_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_fragment_info.json')
        with open(info_path, 'r') as f:
            fragment_info = json.load(f)

        # Ensure coordinates are within bounds
        h, w = labeled.shape
        x = max(0, min(int(x), w - 1))
        y = max(0, min(int(y), h - 1))

        fragment_id = int(labeled[y, x])

        # Check if this fragment is in our info (might be filtered out due to min_size)
        fragment_id_str = str(fragment_id)
        if fragment_id == 0:
            # Clicked on background
            return jsonify({
                'success': True,
                'fragment_id': None,
                'is_background': True
            })

        if fragment_id_str not in fragment_info:
            # Fragment too small, was filtered
            return jsonify({
                'success': True,
                'fragment_id': None,
                'is_tiny': True
            })

        return jsonify({
            'success': True,
            'fragment_id': fragment_id,
            'fragment_info': fragment_info[fragment_id_str]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/fragments/remove', methods=['POST'])
def fragments_remove():
    """Remove selected fragments from the image."""
    data = request.json
    image_id = data.get('image_id')
    fragment_ids = data.get('fragment_ids', [])

    if not image_id:
        return jsonify({'error': 'No image ID provided'}), 400

    if not fragment_ids:
        return jsonify({'error': 'No fragments selected'}), 400

    try:
        # Load labeled array
        labeled_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_labeled.npy')
        if not os.path.exists(labeled_path):
            return jsonify({'error': 'Fragments not detected'}), 404

        labeled = np.load(labeled_path)

        # Load the current output image
        adjusted_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_adjusted.png')
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')
        path = adjusted_path if os.path.exists(adjusted_path) else output_path

        if not os.path.exists(path):
            return jsonify({'error': 'Output image not found'}), 404

        img = Image.open(path)

        # Remove fragments
        img = remove_fragments(img, labeled, fragment_ids)

        # Save the result
        img.save(adjusted_path, 'PNG')

        return jsonify({
            'success': True,
            'image_id': image_id,
            'removed_count': len(fragment_ids)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/fragments/overlay/<image_id>')
def fragments_overlay(image_id):
    """Get the fragment overlay image."""
    selected_remove = request.args.get('remove', '')
    selected_remove = set(int(x) for x in selected_remove.split(',') if x)

    try:
        labeled_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_labeled.npy')
        if not os.path.exists(labeled_path):
            return jsonify({'error': 'Fragments not detected'}), 404

        labeled = np.load(labeled_path)

        # Load fragment info
        info_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_fragment_info.json')
        with open(info_path, 'r') as f:
            fragment_info = {int(k): v for k, v in json.load(f).items()}

        # Create overlay
        overlay = create_fragment_overlay(labeled, fragment_info, selected_remove)

        buffer = BytesIO()
        overlay.save(buffer, 'PNG')
        buffer.seek(0)
        return send_file(buffer, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/image/<image_type>/<image_id>')
def get_image(image_type, image_id):
    """Serve images (original, output, adjusted, or white background versions)."""
    if image_type == 'original':
        path = os.path.join(app.config['UPLOAD_FOLDER'], f'{image_id}_original.png')
    elif image_type == 'output':
        path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')
    elif image_type == 'adjusted':
        path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_adjusted.png')
        # Fall back to output if adjusted doesn't exist
        if not os.path.exists(path):
            path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')
    elif image_type == 'output-white':
        # Output with soft edges, drop shadow, on white background
        path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')
        if not os.path.exists(path):
            return jsonify({'error': 'Image not found'}), 404
        img = Image.open(path)
        if img.mode == 'RGBA':
            img = apply_drop_shadow(img, 2)
            img = prepare_ecommerce_image(img)
        buffer = BytesIO()
        img.save(buffer, 'PNG')
        buffer.seek(0)
        return send_file(buffer, mimetype='image/png')
    elif image_type == 'adjusted-white':
        # Adjusted with soft edges, drop shadow, on white background
        zoom = int(request.args.get('zoom', 100))
        adjusted_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_adjusted.png')
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')
        path = adjusted_path if os.path.exists(adjusted_path) else output_path
        if not os.path.exists(path):
            return jsonify({'error': 'Image not found'}), 404
        img = Image.open(path)
        if img.mode == 'RGBA':
            img = fit_subject_to_square(img, 2048, zoom, max_scale=2.0)
            img = apply_drop_shadow(img, 2)
            img = prepare_ecommerce_image(img)
        buffer = BytesIO()
        img.save(buffer, 'PNG')
        buffer.seek(0)
        return send_file(buffer, mimetype='image/png')
    elif image_type == 'original-white':
        # Original image (no BG removal) with watermark only
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')
        if not os.path.exists(output_path):
            return jsonify({'error': 'Image not found'}), 404
        img = Image.open(output_path)
        if img.mode == 'RGB':
            img = add_watermark(img)
        else:
            # RGBA means BG was removed — fall through to normal pipeline
            img = fit_subject_to_square(img, 2048, int(request.args.get('zoom', 100)), max_scale=2.0)
            img = apply_drop_shadow(img, 2)
            img = prepare_ecommerce_image(img)
        buffer = BytesIO()
        img.save(buffer, 'PNG')
        buffer.seek(0)
        return send_file(buffer, mimetype='image/png')
    else:
        return jsonify({'error': 'Invalid image type'}), 400

    if not os.path.exists(path):
        return jsonify({'error': 'Image not found'}), 404

    return send_file(path, mimetype='image/png')


@app.route('/download/<image_id>')
def download(image_id):
    """Download the processed image."""
    background = request.args.get('background', 'transparent')
    zoom = int(request.args.get('zoom', 100))

    # Check for adjusted version first, then output
    adjusted_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_adjusted.png')
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')

    path = adjusted_path if os.path.exists(adjusted_path) else output_path

    if not os.path.exists(path):
        return jsonify({'error': 'Image not found'}), 404

    img = Image.open(path)

    if background == 'original':
        # Original image with watermark only (no BG removal pipeline)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = add_watermark(img)
        buffer = BytesIO()
        img.save(buffer, 'PNG')
        buffer.seek(0)
        return send_file(buffer, mimetype='image/png', as_attachment=True,
                        download_name=f'original_{image_id}.png')

    if background == 'white':
        # Create e-commerce ready square with white background
        if img.mode == 'RGBA':
            img = fit_subject_to_square(img, 2048, zoom)
            img = apply_drop_shadow(img, 2)
            img = prepare_ecommerce_image(img)

        buffer = BytesIO()
        img.save(buffer, 'PNG')
        buffer.seek(0)
        return send_file(buffer, mimetype='image/png', as_attachment=True,
                        download_name=f'background_removed_{image_id}.png')

    # Transparent download - fit subject to square
    if img.mode == 'RGBA':
        img = fit_subject_to_square(img, 2048, zoom)

    buffer = BytesIO()
    img.save(buffer, 'PNG')
    buffer.seek(0)
    return send_file(buffer, mimetype='image/png', as_attachment=True,
                    download_name=f'background_removed_{image_id}.png')


# ============================================================================
# Color Segment Selection Routes
# ============================================================================

@app.route('/select_color', methods=['POST'])
def select_color():
    """Select a segment at the clicked point using MobileSAM. Accumulates with previous selections."""
    global sam_image_cache

    data = request.json
    image_id = data.get('image_id')
    click_x = data.get('x', 0)
    click_y = data.get('y', 0)
    accumulate = data.get('accumulate', True)  # Add to existing selection by default
    is_foreground = data.get('is_foreground', True)  # True=add to selection, False=subtract
    mode = data.get('mode', 'processed')  # 'processed' or 'original'
    select_mode = data.get('select_mode', 'broad')  # 'broad' or 'detail'
    zoom = data.get('zoom', 100)  # zoom level used in fit_subject_to_square

    if not image_id:
        return jsonify({'error': 'No image ID provided'}), 400

    try:
        # Load the current output image (use original for SAM, it works better)
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{image_id}_original.png')
        adjusted_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_adjusted.png')
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')

        # For display sizing, use the appropriate image based on mode
        if mode == 'original':
            display_path = original_path
        else:
            display_path = adjusted_path if os.path.exists(adjusted_path) else output_path

        if not os.path.exists(display_path):
            return jsonify({'error': 'Image not found'}), 404

        display_img = Image.open(display_path)
        display_size = display_img.size

        # For processed mode, click coords are in 2048x2048 square space.
        # We need to map them back to the original image coordinate space.
        sq_transform = None
        if mode != 'original':
            sq_transform = get_square_transform(display_img, 2048, zoom, max_scale=2.0)

        if sq_transform:
            # Map click from 2048 square space → original image space
            click_x = (click_x - sq_transform['paste_x']) / sq_transform['scale'] + sq_transform['x_min']
            click_y = (click_y - sq_transform['paste_y']) / sq_transform['scale'] + sq_transform['y_min']

        # For SAM, use original image (better segmentation on full detail)
        if os.path.exists(original_path):
            sam_img = Image.open(original_path).convert('RGB')
        else:
            sam_img = display_img.convert('RGB')

        sam_array = np.array(sam_img)
        sam_h, sam_w = sam_array.shape[:2]

        # Scale click coordinates if image sizes differ
        scale_x = sam_w / display_size[0]
        scale_y = sam_h / display_size[1]
        sam_click_x = int(click_x * scale_x)
        sam_click_y = int(click_y * scale_y)

        # Set image in SAM predictor (cache to avoid re-encoding)
        if sam_image_cache['image_id'] != image_id:
            sam_predictor.set_image(sam_array)
            sam_image_cache['image_id'] = image_id
            sam_image_cache['predictor_ready'] = True

        # Predict segment at click point
        point_coords = np.array([[sam_click_x, sam_click_y]])
        point_labels = np.array([1])  # 1 = foreground point

        masks, scores, _ = sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )

        # SAM returns 3 masks at different granularities (small, medium, large).
        if select_mode == 'detail':
            # Detail mode: pick the smallest valid mask that contains the click point.
            # Ideal for small regions like gaps between arms.
            best_idx = None
            best_area = float('inf')
            for i in range(len(masks)):
                m = masks[i]
                if sam_click_y < m.shape[0] and sam_click_x < m.shape[1] and m[sam_click_y, sam_click_x]:
                    area = np.sum(m)
                    if area < best_area and area > 0:
                        best_area = area
                        best_idx = i
            # Fallback to highest score if no mask contains the click point
            if best_idx is None:
                best_idx = np.argmax(scores)
        else:
            # Broad mode (default): pick the highest-scored mask.
            # Best for selecting large regions like full background or big objects.
            best_idx = np.argmax(scores)

        new_mask = (masks[best_idx] * 255).astype(np.uint8)

        # Resize mask to match display image size if needed
        if (sam_w, sam_h) != display_size:
            new_mask = cv2.resize(new_mask, display_size, interpolation=cv2.INTER_NEAREST)

        if new_mask is None or np.sum(new_mask > 0) == 0:
            return jsonify({'success': False, 'message': 'No segment found at click point'})

        # Load existing mask if accumulating
        mask_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_selection_mask.npy')
        if accumulate and os.path.exists(mask_path):
            existing_mask = np.load(mask_path)
            if is_foreground:
                # Add to selection (OR operation)
                combined_mask = np.maximum(existing_mask, new_mask)
            else:
                # Subtract from selection
                combined_mask = np.where(new_mask > 0, 0, existing_mask).astype(np.uint8)
        else:
            combined_mask = new_mask

        # Save the combined mask
        np.save(mask_path, combined_mask)

        # Find contours for the combined mask (for display)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_points = []
        for contour in contours:
            epsilon = 2
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) > 2:
                points = approx.reshape(-1, 2).tolist()
                # Transform contour points from original space → 2048 square space
                if sq_transform:
                    points = [
                        [
                            (p[0] - sq_transform['x_min']) * sq_transform['scale'] + sq_transform['paste_x'],
                            (p[1] - sq_transform['y_min']) * sq_transform['scale'] + sq_transform['paste_y']
                        ]
                        for p in points
                    ]
                contour_points.append(points)

        # Calculate total segment area
        segment_area = int(np.sum(combined_mask > 0))

        return jsonify({
            'success': True,
            'contours': contour_points,
            'area': segment_area,
            'image_id': image_id,
            'score': float(scores[best_idx])
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/remove_selection', methods=['POST'])
def remove_selection():
    """Remove the currently selected color segment."""
    data = request.json
    image_id = data.get('image_id')

    if not image_id:
        return jsonify({'error': 'No image ID provided'}), 400

    try:
        save_undo_backup(image_id)

        # Load the selection mask
        mask_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_selection_mask.npy')
        if not os.path.exists(mask_path):
            return jsonify({'error': 'No selection found. Click on the image first.'}), 404

        mask = np.load(mask_path)

        # Load the current output image
        adjusted_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_adjusted.png')
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')
        path = adjusted_path if os.path.exists(adjusted_path) else output_path

        if not os.path.exists(path):
            return jsonify({'error': 'Image not found'}), 404

        img = Image.open(path)

        # Remove the selected segment
        result = remove_color_segment(img, mask)

        # Smooth jaggies from the binary mask edge
        result = apply_edge_smoothing(result, level=1)

        # Save the adjusted image
        result.save(adjusted_path, 'PNG')

        # Delete the selection mask
        os.remove(mask_path)

        return jsonify({
            'success': True,
            'image_id': image_id
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/clear_selection', methods=['POST'])
def clear_selection():
    """Clear the current selection without removing anything."""
    data = request.json
    image_id = data.get('image_id')

    if not image_id:
        return jsonify({'error': 'No image ID provided'}), 400

    try:
        mask_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_selection_mask.npy')
        if os.path.exists(mask_path):
            os.remove(mask_path)

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/add_selection', methods=['POST'])
def add_selection():
    """Add the selected segment from original image back to the processed result."""
    data = request.json
    image_id = data.get('image_id')

    if not image_id:
        return jsonify({'error': 'No image ID provided'}), 400

    try:
        save_undo_backup(image_id)

        # Load the selection mask
        mask_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_selection_mask.npy')
        if not os.path.exists(mask_path):
            return jsonify({'error': 'No selection found. Click on the original image first.'}), 404

        mask = np.load(mask_path)

        # Load original image
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{image_id}_original.png')
        if not os.path.exists(original_path):
            return jsonify({'error': 'Original image not found'}), 404

        original = Image.open(original_path).convert('RGBA')

        # Load current processed image
        adjusted_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_adjusted.png')
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')
        processed_path = adjusted_path if os.path.exists(adjusted_path) else output_path

        if not os.path.exists(processed_path):
            return jsonify({'error': 'Processed image not found'}), 404

        processed = Image.open(processed_path).convert('RGBA')

        # Resize mask if needed to match processed image size
        if (mask.shape[1], mask.shape[0]) != processed.size:
            mask = cv2.resize(mask, processed.size, interpolation=cv2.INTER_NEAREST)

        # Resize original if needed
        if original.size != processed.size:
            original = original.resize(processed.size, Image.LANCZOS)

        # Dilate and feather the mask for smooth blending (same as remove)
        dilate_pixels = 3
        feather_pixels = 2

        if dilate_pixels > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_pixels * 2 + 1, dilate_pixels * 2 + 1))
            mask = cv2.dilate(mask, kernel, iterations=1)

        # Convert to arrays
        original_arr = np.array(original)
        processed_arr = np.array(processed)

        # Create feathered blend mask
        if feather_pixels > 0:
            mask_float = mask.astype(np.float32) / 255.0
            mask_blurred = cv2.GaussianBlur(mask_float, (0, 0), sigmaX=feather_pixels)
        else:
            mask_blurred = (mask > 0).astype(np.float32)

        # Blend: where mask is set, use original pixels
        for c in range(3):  # RGB channels
            processed_arr[:, :, c] = (
                original_arr[:, :, c] * mask_blurred +
                processed_arr[:, :, c] * (1 - mask_blurred)
            ).astype(np.uint8)

        # Alpha: make selected area fully opaque
        processed_arr[:, :, 3] = np.maximum(
            processed_arr[:, :, 3],
            (mask_blurred * 255).astype(np.uint8)
        )

        result = Image.fromarray(processed_arr, 'RGBA')

        # Save the adjusted image
        result.save(adjusted_path, 'PNG')

        # Delete the selection mask
        os.remove(mask_path)

        return jsonify({
            'success': True,
            'image_id': image_id
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/keep_only_selection', methods=['POST'])
def keep_only_selection():
    """Keep ONLY the selected segment, discard everything else."""
    data = request.json
    image_id = data.get('image_id')
    mode = data.get('mode', 'processed')  # 'original' or 'processed'

    if not image_id:
        return jsonify({'error': 'No image ID provided'}), 400

    try:
        save_undo_backup(image_id)

        # Load the selection mask
        mask_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_selection_mask.npy')
        if not os.path.exists(mask_path):
            return jsonify({'error': 'No selection found. Click on the image first.'}), 404

        mask = np.load(mask_path)

        adjusted_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_adjusted.png')
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')

        if mode == 'original':
            # Load original image
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{image_id}_original.png')
            if not os.path.exists(original_path):
                return jsonify({'error': 'Original image not found'}), 404

            original = Image.open(original_path).convert('RGBA')

            # Load current processed image to work in its coordinate space
            processed_path = adjusted_path if os.path.exists(adjusted_path) else output_path
            if not os.path.exists(processed_path):
                return jsonify({'error': 'Processed image not found'}), 404
            processed = Image.open(processed_path).convert('RGBA')

            # Resize mask and original to match processed output dimensions
            if (mask.shape[1], mask.shape[0]) != processed.size:
                mask = cv2.resize(mask, processed.size, interpolation=cv2.INTER_NEAREST)
            if original.size != processed.size:
                original = original.resize(processed.size, Image.LANCZOS)

            # Dilate mask slightly + feather edges for smooth blend
            dilate_pixels = 3
            feather_pixels = 2

            if dilate_pixels > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_pixels * 2 + 1, dilate_pixels * 2 + 1))
                mask = cv2.dilate(mask, kernel, iterations=1)

            original_arr = np.array(original)

            # Create feathered blend mask
            if feather_pixels > 0:
                mask_float = mask.astype(np.float32) / 255.0
                mask_blurred = cv2.GaussianBlur(mask_float, (0, 0), sigmaX=feather_pixels)
            else:
                mask_blurred = (mask > 0).astype(np.float32)

            # Keep original RGB where mask is set, alpha=0 everywhere else
            result_arr = original_arr.copy()
            result_arr[:, :, 3] = (mask_blurred * 255).astype(np.uint8)

            result = Image.fromarray(result_arr, 'RGBA')

            # Save as adjusted (preserves _output.png for coordinate mapping)
            result.save(adjusted_path, 'PNG')

        else:
            # mode == 'processed'
            path = adjusted_path if os.path.exists(adjusted_path) else output_path
            if not os.path.exists(path):
                return jsonify({'error': 'Processed image not found'}), 404

            img = Image.open(path).convert('RGBA')

            # Resize mask to match processed image dimensions if needed
            if (mask.shape[1], mask.shape[0]) != img.size:
                mask = cv2.resize(mask, img.size, interpolation=cv2.INTER_NEAREST)

            # Dilate mask slightly + feather edges for smooth blend
            dilate_pixels = 3
            feather_pixels = 2

            if dilate_pixels > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_pixels * 2 + 1, dilate_pixels * 2 + 1))
                mask = cv2.dilate(mask, kernel, iterations=1)

            img_arr = np.array(img)

            # Create feathered blend mask
            if feather_pixels > 0:
                mask_float = mask.astype(np.float32) / 255.0
                mask_blurred = cv2.GaussianBlur(mask_float, (0, 0), sigmaX=feather_pixels)
            else:
                mask_blurred = (mask > 0).astype(np.float32)

            # Keep only where mask IS set — zero alpha everywhere else
            img_arr[:, :, 3] = (img_arr[:, :, 3].astype(np.float32) * mask_blurred).astype(np.uint8)

            result = Image.fromarray(img_arr, 'RGBA')
            result = apply_edge_smoothing(result, level=1)

            # Don't recenter — display route handles visual centering via
            # fit_subject_to_square, and shifting pixels here would break
            # the coordinate mapping between processed output and original.
            result.save(adjusted_path, 'PNG')

        # Delete the selection mask
        os.remove(mask_path)

        return jsonify({
            'success': True,
            'image_id': image_id
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset_image():
    """Reset image to original output (undo all adjustments)."""
    data = request.json
    image_id = data.get('image_id')

    if not image_id:
        return jsonify({'error': 'No image ID provided'}), 400

    try:
        # Delete adjusted image if it exists
        adjusted_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_adjusted.png')
        if os.path.exists(adjusted_path):
            os.remove(adjusted_path)
            print(f"Reset: deleted {adjusted_path}")

        return jsonify({
            'success': True,
            'image_id': image_id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/recenter', methods=['POST'])
def recenter():
    """Recenter the subject in the image."""
    data = request.json
    image_id = data.get('image_id')

    if not image_id:
        return jsonify({'error': 'No image ID provided'}), 400

    try:
        # Load from adjusted if exists, otherwise from output
        adjusted_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_adjusted.png')
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')

        if os.path.exists(adjusted_path):
            img = Image.open(adjusted_path)
        elif os.path.exists(output_path):
            img = Image.open(output_path)
        else:
            return jsonify({'error': 'Image not found'}), 404

        # Recenter the subject
        img = recenter_subject(img)

        # Save adjusted image
        img.save(adjusted_path, 'PNG')

        return jsonify({
            'success': True,
            'image_id': image_id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== GoPro API Routes ====================

@app.route('/gopro/status')
def gopro_status():
    """Check GoPro connection status."""
    if not GOPRO_CREDS_FILE.exists() or not GOPRO_CERT_FILE.exists():
        return jsonify({
            'connected': False,
            'available': False,
            'error': 'GoPro not provisioned. Place cohn_credentials.json and cohn.crt in the app folder.'
        })

    try:
        state = gopro_request("/gopro/camera/state", timeout=5, retries=1)
        status = state.get("status", {})
        batt_pct = status.get("70")       # 0-100
        batt_bars = status.get("2")       # 0=empty,1=low,2=med,3=full,4=charging
        charging = batt_bars == 4
        return jsonify({
            'connected': True,
            'available': True,
            'ip_address': gopro_config.get('ip_address', 'unknown'),
            'photo_mode_set': gopro_photo_mode_set,
            'battery_percent': batt_pct,
            'battery_charging': charging
        })
    except Exception as e:
        return jsonify({
            'connected': False,
            'available': True,  # Credentials exist but camera not reachable
            'error': str(e)
        })


# ==================== GoPro Provisioning Routes ====================

class ProvisionLogHandler(logging.Handler):
    """Custom log handler that captures gopro_cohn log messages to update provision_status."""
    STEP_MAP = {
        'Scanning for GoPro': 'connecting_ble',
        'Found GoPro': 'connecting_ble',
        'Connecting to': 'connecting_ble',
        'Connected!': 'connecting_ble',
        'Pairing': 'connecting_ble',
        'BLE connection established': 'connecting_ble',
        'Enabling notifications': 'connecting_ble',
        'Setting camera date/time': 'setting_time',
        'Date/time set': 'setting_time',
        'Scanning for WiFi': 'scanning_wifi',
        'Scan complete': 'scanning_wifi',
        'Retrieving scan results': 'scanning_wifi',
        'Connecting to WiFi': 'connecting_wifi',
        'connecting to new': 'connecting_wifi',
        'Network previously configured': 'connecting_wifi',
        'Successfully connected to': 'connecting_wifi',
        'Provisioning COHN': 'creating_cert',
        'Clearing existing': 'creating_cert',
        'Certificate cleared': 'creating_cert',
        'Creating new COHN': 'creating_cert',
        'Certificate created': 'creating_cert',
        'Retrieving COHN certificate': 'getting_credentials',
        'Certificate retrieved': 'getting_credentials',
        'Waiting for COHN': 'getting_credentials',
        'COHN connected': 'getting_credentials',
        'COHN provisioning complete': 'complete',
    }

    def emit(self, record):
        msg = record.getMessage()
        step = None
        for keyword, step_name in self.STEP_MAP.items():
            if keyword.lower() in msg.lower():
                step = step_name
                break
        with provision_lock:
            if step:
                provision_status['step'] = step
            provision_status['message'] = msg


def _run_provisioning(ssid, password, identifier):
    """Background thread that runs the async provisioning process."""
    global provision_status

    # Set up logging capture
    gopro_logger = logging.getLogger("gopro_cohn")
    gopro_logger.setLevel(logging.DEBUG)
    handler = ProvisionLogHandler()
    gopro_logger.addHandler(handler)

    try:
        with provision_lock:
            provision_status['step'] = 'connecting_ble'
            provision_status['message'] = 'Searching for GoPro via Bluetooth...'
            provision_status['done'] = False
            provision_status['error'] = None

        credentials = asyncio.run(provision_gopro(
            ssid=ssid,
            password=password,
            identifier=identifier or None,
            cert_path=GOPRO_CERT_FILE,
            creds_path=GOPRO_CREDS_FILE,
        ))

        # Verify credentials were saved; save again if files are missing
        if not GOPRO_CERT_FILE.exists() or not GOPRO_CREDS_FILE.exists():
            logging.warning("Credential files not found after provisioning — saving manually")
            credentials.save(GOPRO_CERT_FILE, GOPRO_CREDS_FILE)

        with provision_lock:
            provision_status['step'] = 'complete'
            provision_status['message'] = f'Provisioning complete! IP: {credentials.ip_address}'
            provision_status['done'] = True

        # Reinitialize the GoPro session with new credentials
        init_gopro_session()

        # Reinitialize stream manager with new credentials
        global stream_manager
        stream_manager = StreamManager(GOPRO_CREDS_FILE, GOPRO_CERT_FILE, Path(app.config['GOPRO_PHOTOS']))

    except Exception as e:
        with provision_lock:
            provision_status['step'] = 'error'
            provision_status['message'] = str(e)
            provision_status['done'] = True
            provision_status['error'] = str(e)
    finally:
        gopro_logger.removeHandler(handler)


@app.route('/gopro/provision', methods=['POST'])
def gopro_provision():
    """Start GoPro COHN provisioning."""
    global provision_status

    with provision_lock:
        if provision_status.get('step') and not provision_status.get('done'):
            return jsonify({'error': 'Provisioning already in progress'}), 409

    data = request.json
    ssid = data.get('ssid', '').strip()
    password = data.get('password', '').strip()
    identifier = data.get('identifier', '').strip()

    if not ssid or not password:
        return jsonify({'error': 'WiFi SSID and password are required'}), 400

    # Reset status
    with provision_lock:
        provision_status = {
            'step': 'connecting_ble',
            'message': 'Starting provisioning...',
            'done': False,
            'error': None
        }

    # Start background thread
    thread = threading.Thread(target=_run_provisioning, args=(ssid, password, identifier), daemon=True)
    thread.start()

    return jsonify({'success': True, 'message': 'Provisioning started'})


@app.route('/gopro/provision/status')
def gopro_provision_status():
    """SSE endpoint for provisioning progress."""
    def generate():
        last_msg = None
        while True:
            with provision_lock:
                status = dict(provision_status)

            msg = json.dumps(status)
            if msg != last_msg:
                yield f"data: {msg}\n\n"
                last_msg = msg

            if status.get('done'):
                break

            time.sleep(0.5)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/gopro/forget', methods=['POST'])
def gopro_forget():
    """Remove GoPro pairing credentials."""
    global gopro_session, gopro_config, gopro_photo_mode_set, stream_manager

    try:
        if GOPRO_CREDS_FILE.exists():
            os.remove(GOPRO_CREDS_FILE)
        if GOPRO_CERT_FILE.exists():
            os.remove(GOPRO_CERT_FILE)

        gopro_session = None
        gopro_config = None
        gopro_photo_mode_set = False

        # Reinitialize stream manager (will be non-functional without creds)
        stream_manager = StreamManager(GOPRO_CREDS_FILE, GOPRO_CERT_FILE, Path(app.config['GOPRO_PHOTOS']))

        return jsonify({'success': True, 'message': 'GoPro pairing removed'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/settings')
def get_settings():
    """Return server-side settings as JSON."""
    paired = GOPRO_CREDS_FILE.exists() and GOPRO_CERT_FILE.exists()
    ip = None
    if paired:
        try:
            with open(GOPRO_CREDS_FILE) as f:
                creds = json.load(f)
            ip = creds.get('ip_address')
        except Exception:
            pass

    return jsonify({
        'gopro': {
            'paired': paired,
            'ip_address': ip,
        }
    })


@app.route('/settings', methods=['POST'])
def save_settings():
    """Save settings (reserved for future server-side settings)."""
    return jsonify({'success': True})


@app.route('/gopro/preview', methods=['POST'])
def gopro_preview():
    """Take a quick preview photo and return URL."""
    global gopro_photo_in_progress, gopro_last_preview

    if gopro_photo_in_progress:
        return jsonify({'success': False, 'error': 'Camera busy'})

    try:
        gopro_photo_in_progress = True

        # Ensure photo mode (sets once per session)
        gopro_ensure_photo_mode_once()

        # Take the photo
        gopro_request("/gopro/camera/shutter/start", timeout=10)
        time.sleep(1.0)

        # Get latest photo
        folder, filename = gopro_get_latest_photo()
        if not filename:
            return jsonify({'success': False, 'error': 'No photo found'})

        # Download it
        file_path = f"{folder}/{filename}"
        local_path = gopro_download_media(file_path)

        # Track this preview for potential capture use
        gopro_last_preview = {
            'path': local_path,
            'filename': filename,
            'time': time.time()
        }

        return jsonify({
            'success': True,
            'photo_url': f'/gopro/photo/{filename}'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    finally:
        gopro_photo_in_progress = False


@app.route('/gopro/photo/<filename>')
def gopro_photo(filename):
    """Serve a GoPro photo."""
    photo_path = Path(app.config['GOPRO_PHOTOS']) / filename
    if photo_path.exists():
        return send_file(photo_path, mimetype='image/jpeg')
    return jsonify({'error': 'Photo not found'}), 404


def recenter_subject(img):
    """
    Find the bounding box of the subject (non-transparent area) and center it.
    """
    if img.mode != 'RGBA':
        return img

    data = np.array(img)
    alpha = data[:, :, 3]

    # Find rows and columns with non-transparent pixels
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)

    if not np.any(rows) or not np.any(cols):
        return img  # No subject found

    # Get bounding box
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]
    min_row, max_row = row_indices[0], row_indices[-1]
    min_col, max_col = col_indices[0], col_indices[-1]

    # Calculate subject dimensions and center
    subject_height = max_row - min_row + 1
    subject_width = max_col - min_col + 1
    subject_center_y = (min_row + max_row) // 2
    subject_center_x = (min_col + max_col) // 2

    # Calculate image center
    height, width = alpha.shape
    image_center_y = height // 2
    image_center_x = width // 2

    # Calculate offset needed to center
    offset_y = image_center_y - subject_center_y
    offset_x = image_center_x - subject_center_x

    if offset_x == 0 and offset_y == 0:
        return img  # Already centered

    # Create new image and paste with offset
    result = Image.new('RGBA', (width, height), (255, 255, 255, 0))

    # Crop the subject area
    subject = img.crop((min_col, min_row, max_col + 1, max_row + 1))

    # Calculate new position (centered)
    new_x = (width - subject_width) // 2
    new_y = (height - subject_height) // 2

    result.paste(subject, (new_x, new_y), subject)

    return result


def apply_subject_zoom(img, zoom_percent):
    """
    Scale up the subject (non-transparent area) within the canvas.
    The subject is enlarged and centered, keeping canvas size the same.
    zoom_percent is 100-400 (100 = no zoom, 200 = 2x larger subject).
    """
    if zoom_percent <= 100 or img.mode != 'RGBA':
        return img

    width, height = img.size
    scale = zoom_percent / 100

    # Calculate new size for the scaled content
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Scale up the image
    scaled = img.resize((new_width, new_height), Image.LANCZOS)

    # Create new canvas at original size
    result = Image.new('RGBA', (width, height), (255, 255, 255, 0))

    # Calculate position to center the scaled image
    paste_x = (width - new_width) // 2
    paste_y = (height - new_height) // 2

    # Paste the scaled image (will be cropped to canvas bounds)
    result.paste(scaled, (paste_x, paste_y), scaled)

    return result


def apply_color_correction(img, temperature, tint, brightness=0):
    """
    Apply color temperature, tint, and brightness correction to an image.

    temperature: -100 to +100
        - Negative = cooler (more blue)
        - Positive = warmer (more orange/yellow)

    tint: -100 to +100
        - Negative = more green
        - Positive = more magenta

    brightness: -100 to +100
        - Negative = darker
        - Positive = brighter
    """
    import numpy as np

    if temperature == 0 and tint == 0 and brightness == 0:
        return img

    # Preserve alpha channel if present
    has_alpha = img.mode == 'RGBA'
    if has_alpha:
        alpha = img.split()[3]
        rgb_img = img.convert('RGB')
    else:
        rgb_img = img.convert('RGB')

    # Convert to numpy array
    arr = np.array(rgb_img, dtype=np.float32)

    # Apply brightness adjustment first
    if brightness != 0:
        brightness_factor = brightness / 100.0  # -1 to +1
        # Scale brightness effect (max ~50 units change)
        arr = arr + (brightness_factor * 50)
        arr = np.clip(arr, 0, 255)

    # Apply temperature adjustment
    # Warm = more red/yellow, less blue
    # Cool = less red, more blue
    if temperature != 0:
        temp_factor = temperature / 100.0  # -1 to +1

        # Adjust red channel (increase for warm, decrease for cool)
        arr[:, :, 0] = np.clip(arr[:, :, 0] + (temp_factor * 30), 0, 255)

        # Adjust blue channel (decrease for warm, increase for cool)
        arr[:, :, 2] = np.clip(arr[:, :, 2] - (temp_factor * 30), 0, 255)

        # Slight yellow/orange shift for warm (add to green slightly)
        if temp_factor > 0:
            arr[:, :, 1] = np.clip(arr[:, :, 1] + (temp_factor * 10), 0, 255)

    # Apply tint adjustment
    # Green tint = more green
    # Magenta tint = less green (or more red+blue)
    if tint != 0:
        tint_factor = tint / 100.0  # -1 to +1

        # Adjust green channel
        arr[:, :, 1] = np.clip(arr[:, :, 1] - (tint_factor * 25), 0, 255)

        # Slight compensation on red/blue for magenta
        if tint_factor > 0:
            arr[:, :, 0] = np.clip(arr[:, :, 0] + (tint_factor * 10), 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] + (tint_factor * 10), 0, 255)

    # Convert back to PIL Image
    result = Image.fromarray(arr.astype(np.uint8), 'RGB')

    # Restore alpha channel if present
    if has_alpha:
        result = result.convert('RGBA')
        result.putalpha(alpha)

    return result


def apply_edge_smoothing(img, level=2):
    """
    Smooth edges using Gaussian blur on alpha channel.
    Only smooths existing semi-transparent edge pixels without spreading
    the alpha outward (which would cause grey fringing on white backgrounds).

    level: 1=subtle, 2=medium, 3=strong
    """
    from PIL import ImageFilter
    import numpy as np

    if img.mode != 'RGBA':
        return img

    # Split channels
    r, g, b, a = img.split()

    # Use a very light blur - just enough for anti-aliasing, not enough to spread
    blur_radius = 0.3 + (level * 0.2)  # 0.5, 0.7, 0.9 for levels 1, 2, 3
    smoothed_alpha = a.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Clamp: don't let smoothing expand the alpha beyond the original extent.
    # This prevents grey fringing on edges.
    a_arr = np.array(a)
    s_arr = np.array(smoothed_alpha)
    # Where original alpha was 0, keep it 0 (don't bleed outward)
    s_arr = np.where(a_arr == 0, 0, s_arr)
    smoothed_alpha = Image.fromarray(s_arr)

    # Merge channels back
    return Image.merge('RGBA', (r, g, b, smoothed_alpha))


def correct_fisheye(img, strength=0.3):
    """
    Correct barrel/fisheye distortion from wide-angle GoPro lens.

    strength: 0.0 = no correction, 0.5 = strong correction
    """
    import numpy as np
    import cv2

    # Convert PIL to OpenCV format
    if img.mode == 'RGBA':
        img_array = np.array(img)
        # OpenCV uses BGR, PIL uses RGB
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)
    else:
        img_array = np.array(img.convert('RGB'))
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    h, w = img_cv.shape[:2]

    # Camera matrix (approximate for GoPro-like wide angle)
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    # Distortion coefficients for barrel distortion correction
    # k1 negative = correct barrel distortion (pincushion correction)
    k1 = -strength  # Primary radial distortion
    k2 = strength * 0.1  # Secondary radial
    dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float32)

    # Get optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    # Undistort
    corrected = cv2.undistort(img_cv, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Crop to ROI if valid
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        corrected = corrected[y:y+rh, x:x+rw]
        # Resize back to original dimensions
        corrected = cv2.resize(corrected, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Convert back to PIL
    if img.mode == 'RGBA':
        corrected = cv2.cvtColor(corrected, cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(corrected, 'RGBA')
    else:
        corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
        return Image.fromarray(corrected, 'RGB')


def apply_drop_shadow(img, level):
    """
    Apply a drop shadow to the image.
    Light source is from bottom-left, so shadow falls to top-right.

    level: 1=soft, 2=medium, 3=strong
    """
    from PIL import ImageFilter
    import numpy as np
    import cv2

    if img.mode != 'RGBA':
        return img

    width, height = img.size

    # Shadow parameters based on level
    # Large blur = smooth gradient into background (no hard grey edge)
    # Negative offset_y = shadow falls upward (light from bottom-left)
    params = {
        1: {'offset': (4, -4), 'blur': 14, 'opacity': 0.08},
        2: {'offset': (6, -6), 'blur': 20, 'opacity': 0.12},
        3: {'offset': (10, -10), 'blur': 26, 'opacity': 0.18},
    }.get(level, {'offset': (6, -6), 'blur': 20, 'opacity': 0.12})

    offset_x, offset_y = params['offset']
    blur_radius = params['blur']
    opacity = params['opacity']

    # Erode alpha so shadow starts away from the subject edge (prevents grey halo)
    alpha = img.split()[3]
    alpha_arr = np.array(alpha)
    erode_kernel = np.ones((3, 3), np.uint8)
    alpha_eroded = cv2.erode(alpha_arr, erode_kernel, iterations=2)
    shadow_alpha = Image.fromarray(alpha_eroded)

    # Build canvas large enough for shadow in any direction + blur padding
    abs_dx = abs(offset_x)
    abs_dy = abs(offset_y)
    canvas_w = width + abs_dx + blur_radius * 2
    canvas_h = height + abs_dy + blur_radius * 2

    # Image origin on canvas (room for blur + negative offsets)
    img_x = blur_radius + max(0, -offset_x)
    img_y = blur_radius + max(0, -offset_y)

    # Shadow origin = image origin + offset
    shadow_x = img_x + offset_x
    shadow_y = img_y + offset_y

    shadow = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
    shadow.paste((0, 0, 0, int(255 * opacity)), (shadow_x, shadow_y), shadow_alpha)

    # Blur the shadow for smooth gradient falloff
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Create result canvas
    result = Image.new('RGBA', (width, height), (255, 255, 255, 0))

    # Crop shadow back to original image bounds
    shadow_cropped = shadow.crop((img_x, img_y, img_x + width, img_y + height))
    result = Image.alpha_composite(result, shadow_cropped)

    # Paste original image on top
    result = Image.alpha_composite(result, img)

    return result


# ==================== GoPro Stream Routes ====================

@app.route('/gopro/stream/start', methods=['POST'])
def gopro_stream_start():
    """Start the GoPro preview stream."""
    try:
        success = stream_manager.start_stream()
        status = stream_manager.get_status()
        return jsonify({
            'success': success,
            **status
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/gopro/stream/stop', methods=['POST'])
def gopro_stream_stop():
    """Stop the GoPro preview stream."""
    try:
        stream_manager.stop_stream()
        return jsonify({'success': True, 'status': 'stopped'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/gopro/stream/status')
def gopro_stream_status():
    """Get current stream status."""
    return jsonify(stream_manager.get_status())


@app.route('/gopro/stream/feed')
def gopro_stream_feed():
    """MJPEG stream endpoint for browser display."""
    def generate():
        last_frame_id = 0
        wait_count = 0

        while stream_manager.streaming:
            frame = stream_manager.get_frame()
            current_count = stream_manager.frame_count

            if frame and current_count > last_frame_id:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                last_frame_id = current_count
                wait_count = 0
            else:
                wait_count += 1
                if wait_count > 300:  # ~30 seconds timeout
                    break

            time.sleep(0.1)

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/gopro/stream/snapshot', methods=['POST'])
def gopro_stream_snapshot():
    """Capture current frame and process for background removal."""
    global gopro_last_preview

    timings = {}
    total_start = time.time()


    # Get snapshot from stream
    step_start = time.time()
    snapshot_path = stream_manager.capture_snapshot()
    if not snapshot_path:
        return jsonify({'error': 'No frame available. Is the stream running?'}), 400
    timings['snapshot'] = round((time.time() - step_start) * 1000)

    try:
        # Generate unique ID for this image
        image_id = str(uuid.uuid4())

        # Read and process image
        step_start = time.time()
        img = Image.open(snapshot_path)
        print(f"Stream snapshot captured: {img.size[0]}x{img.size[1]}")
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        timings['read'] = round((time.time() - step_start) * 1000)

        # Apply fisheye/lens distortion correction
        step_start = time.time()
        img = correct_fisheye(img, strength=0.25)
        timings['lens_correct'] = round((time.time() - step_start) * 1000)

        # Store original size
        original_size = img.size

        # Resize if needed
        step_start = time.time()
        img, was_resized = resize_image(img)
        timings['resize'] = round((time.time() - step_start) * 1000)

        # Save original for comparison
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{image_id}_original.png')
        img.save(original_path, 'PNG')

        # Remove background using hybrid method (AI + green cleanup)
        step_start = time.time()
        model = request.form.get('model', 'u2netp')
        output = remove_background_ai(img, model=model)
        timings['method'] = model
        timings['remove_bg'] = round((time.time() - step_start) * 1000)

        # Skip edge erosion to preserve thin details like straps
        timings['clean_edges'] = 0

        # Save processed image
        step_start = time.time()
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')
        output.save(output_path, 'PNG')
        timings['save'] = round((time.time() - step_start) * 1000)

        timings['total'] = round((time.time() - total_start) * 1000)

        return jsonify({
            'success': True,
            'image_id': image_id,
            'timings': timings,
            'original_size': list(original_size),
            'processed_size': list(img.size),
            'was_resized': was_resized,
            'source': 'stream'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/gopro/zoom', methods=['POST'])
def gopro_zoom():
    """Set the GoPro digital zoom level."""
    data = request.json
    percent = data.get('percent', 0)

    try:
        percent = int(percent)
        percent = max(0, min(100, percent))

        success = stream_manager.set_zoom(percent)
        return jsonify({'success': success, 'zoom': percent})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/gopro/white_balance', methods=['POST'])
def gopro_white_balance():
    """Set the GoPro white balance setting."""
    data = request.json
    option = data.get('option', 0)

    try:
        option = int(option)
        # White balance setting ID is 22
        # Options: 0=Auto, 1=2300K, 2=2800K, 3=3200K, 4=4000K, 5=4500K, 6=5000K, 7=5500K, 8=6000K, 9=6500K, 10=Native
        success = stream_manager.set_setting(22, option)
        return jsonify({'success': success, 'white_balance': option})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/gopro/capture', methods=['POST'])
def gopro_capture():
    """Capture a photo with GoPro and process it for background removal."""
    global gopro_photo_in_progress, gopro_last_preview

    timings = {}
    total_start = time.time()
    used_preview = False


    # Check if stream is running - use snapshot instead
    if stream_manager.streaming:
        # Redirect to stream snapshot for instant capture
        return gopro_stream_snapshot()

    # Check if we have a recent preview we can use (within 15 seconds)
    if gopro_last_preview and (time.time() - gopro_last_preview['time']) < 15:
        local_path = gopro_last_preview['path']
        if local_path.exists():
            print(f"Using recent preview image: {gopro_last_preview['filename']}")
            used_preview = True
            timings['setup'] = 0
            timings['capture'] = 0
            timings['find'] = 0
            timings['download'] = 0

    if not used_preview:
        if gopro_photo_in_progress:
            return jsonify({'error': 'Photo capture already in progress'}), 409

        try:
            gopro_photo_in_progress = True

            # Ensure photo mode (sets once per session)
            step_start = time.time()
            gopro_ensure_photo_mode_once()
            timings['setup'] = round((time.time() - step_start) * 1000)

            # Take the photo
            step_start = time.time()
            gopro_request("/gopro/camera/shutter/start", timeout=10)
            time.sleep(1.0)  # Wait for photo to be saved
            timings['capture'] = round((time.time() - step_start) * 1000)

            # Get latest photo
            step_start = time.time()
            folder, filename = gopro_get_latest_photo()
            if not filename:
                return jsonify({'error': 'No photo found on camera'}), 500
            timings['find'] = round((time.time() - step_start) * 1000)

            # Download it
            step_start = time.time()
            file_path = f"{folder}/{filename}"
            local_path = gopro_download_media(file_path)
            timings['download'] = round((time.time() - step_start) * 1000)

        finally:
            gopro_photo_in_progress = False

    try:

        # Generate unique ID for this image
        image_id = str(uuid.uuid4())

        # Read and process image (same as upload flow)
        step_start = time.time()
        img = Image.open(local_path)
        if used_preview:
            print(f"Using preview image: {img.size[0]}x{img.size[1]}")
        else:
            print(f"GoPro photo captured: {img.size[0]}x{img.size[1]}")
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        timings['read'] = round((time.time() - step_start) * 1000)

        # Store original size for mask corrections
        original_size = img.size

        # Resize if needed
        step_start = time.time()
        img, was_resized = resize_image(img)
        timings['resize'] = round((time.time() - step_start) * 1000)

        if was_resized:
            print(f"Resized for processing: {img.size[0]}x{img.size[1]}")
        else:
            print(f"No resize needed (already <= {MAX_SIZE}px)")

        # Save original for comparison
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{image_id}_original.png')
        img.save(original_path, 'PNG')

        # Remove background using hybrid method (AI + green cleanup)
        step_start = time.time()
        model = request.form.get('model', 'u2netp')
        output = remove_background_ai(img, model=model)
        timings['method'] = model
        timings['remove_bg'] = round((time.time() - step_start) * 1000)

        # Skip edge erosion to preserve thin details like straps
        timings['clean_edges'] = 0

        # Save processed image
        step_start = time.time()
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')
        output.save(output_path, 'PNG')
        timings['save'] = round((time.time() - step_start) * 1000)

        timings['total'] = round((time.time() - total_start) * 1000)

        # Clear the used preview
        if used_preview:
            gopro_last_preview = None

        return jsonify({
            'success': True,
            'image_id': image_id,
            'timings': timings,
            'original_size': list(original_size),
            'processed_size': list(img.size),
            'was_resized': was_resized,
            'used_preview': used_preview
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Load models now if not already loaded (handles non-debug or first run)
    _load_models()

    # Try to initialize GoPro session on startup
    if GOPRO_CREDS_FILE.exists() and GOPRO_CERT_FILE.exists():
        print("GoPro credentials found. Initializing connection...")
        success, msg = init_gopro_session()
        if success:
            print(f"GoPro connected at {gopro_config.get('ip_address', 'unknown')}")
        else:
            print(f"GoPro initialization failed: {msg}")
    else:
        print("GoPro credentials not found. GoPro capture disabled.")
        print("To enable, place cohn_credentials.json and cohn.crt in the app folder.")

    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
