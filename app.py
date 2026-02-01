import os
import time
import uuid
import json
import threading
from io import BytesIO
from pathlib import Path
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from rembg import remove, new_session
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter
import requests

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(__file__), 'outputs')
app.config['GOPRO_PHOTOS'] = os.path.join(os.path.dirname(__file__), 'gopro_photos')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['GOPRO_PHOTOS'], exist_ok=True)

# Pre-load the model on startup for faster processing
print("Loading u2netp model... This may take a moment on first run.")
session = new_session("u2netp")
print("Model loaded successfully!")

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


def prepare_ecommerce_image(img, watermark_scale=0.075, watermark_opacity=0.3, watermark_margin=15):
    """Prepare image for e-commerce: white background, watermark."""
    # Effects disabled for evaluation - just composite on white background

    img_data = np.array(img)
    h, w = img_data.shape[:2]

    # Start with white background
    result = np.full((h, w, 3), 255, dtype=np.uint8)

    # Composite the image directly on white (no shadow, no edge effects)
    alpha = img_data[:, :, 3].astype(float) / 255.0
    for c in range(3):
        result[:, :, c] = (img_data[:, :, c] * alpha + result[:, :, c] * (1 - alpha)).astype(np.uint8)

    result_img = Image.fromarray(result, 'RGB')

    # Add watermark
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
        result_img = result_img.convert('RGBA')
        result_img.paste(wm_resized, (wm_x, wm_y), wm_resized)
        result_img = result_img.convert('RGB')

    return result_img


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

        # Store original size for mask corrections
        original_size = img.size

        # Resize if needed
        step_start = time.time()
        img, was_resized = resize_image(img)
        timings['resize'] = round((time.time() - step_start) * 1000)

        # Save original for comparison
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{image_id}_original.png')
        img.save(original_path, 'PNG')

        # Remove background
        step_start = time.time()
        output = remove(img, session=session)
        timings['remove_bg'] = round((time.time() - step_start) * 1000)

        # Clean edges - erode to remove contaminated pixels
        step_start = time.time()
        output = clean_edges(output, erode_pixels=3)
        timings['clean_edges'] = round((time.time() - step_start) * 1000)

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


@app.route('/adjust', methods=['POST'])
def adjust():
    """Adjust the processed image with threshold, mask corrections, or zoom."""
    data = request.json
    image_id = data.get('image_id')
    threshold = data.get('threshold', 50)
    mask_data = data.get('mask_data')
    original_size = data.get('original_size', [0, 0])
    subject_zoom = data.get('subject_zoom', 100)  # 100-400%

    if not image_id:
        return jsonify({'error': 'No image ID provided'}), 400

    try:
        # Load from adjusted if exists, otherwise from output (cumulative edits)
        adjusted_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_adjusted.png')
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')

        if os.path.exists(adjusted_path):
            img = Image.open(adjusted_path)
        elif os.path.exists(output_path):
            img = Image.open(output_path)
        else:
            return jsonify({'error': 'Image not found'}), 404

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
            img = prepare_ecommerce_image(img)
        buffer = BytesIO()
        img.save(buffer, 'PNG')
        buffer.seek(0)
        return send_file(buffer, mimetype='image/png')
    elif image_type == 'adjusted-white':
        # Adjusted with soft edges, drop shadow, on white background
        adjusted_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_adjusted.png')
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')
        path = adjusted_path if os.path.exists(adjusted_path) else output_path
        if not os.path.exists(path):
            return jsonify({'error': 'Image not found'}), 404
        img = Image.open(path)
        if img.mode == 'RGBA':
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

    # Check for adjusted version first, then output
    adjusted_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_adjusted.png')
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{image_id}_output.png')

    path = adjusted_path if os.path.exists(adjusted_path) else output_path

    if not os.path.exists(path):
        return jsonify({'error': 'Image not found'}), 404

    img = Image.open(path)

    if background == 'white':
        # Create e-commerce ready version with soft edges, shadow, and watermark
        if img.mode == 'RGBA':
            img = prepare_ecommerce_image(img)

        buffer = BytesIO()
        img.save(buffer, 'PNG')
        buffer.seek(0)
        return send_file(buffer, mimetype='image/png', as_attachment=True,
                        download_name=f'background_removed_{image_id}.png')

    return send_file(path, mimetype='image/png', as_attachment=True,
                    download_name=f'background_removed_{image_id}.png')


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
        return jsonify({
            'connected': True,
            'available': True,
            'ip_address': gopro_config.get('ip_address', 'unknown'),
            'photo_mode_set': gopro_photo_mode_set
        })
    except Exception as e:
        return jsonify({
            'connected': False,
            'available': True,  # Credentials exist but camera not reachable
            'error': str(e)
        })


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


@app.route('/gopro/capture', methods=['POST'])
def gopro_capture():
    """Capture a photo with GoPro and process it for background removal."""
    global gopro_photo_in_progress, gopro_last_preview

    timings = {}
    total_start = time.time()
    used_preview = False

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

        # Remove background
        step_start = time.time()
        output = remove(img, session=session)
        timings['remove_bg'] = round((time.time() - step_start) * 1000)

        # Clean edges - erode to remove contaminated pixels
        step_start = time.time()
        output = clean_edges(output, erode_pixels=3)
        timings['clean_edges'] = round((time.time() - step_start) * 1000)

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
