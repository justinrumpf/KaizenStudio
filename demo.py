"""
KaizenStudio Demo - Background Removal & Image Editing Tools
============================================================
A standalone demo to test and validate background removal and image editing functions.

Run with: python demo.py
Then open: http://localhost:5001
"""

import os
import uuid
import time
from io import BytesIO
from PIL import Image, ImageFilter
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template_string
from rembg import remove, new_session
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation, distance_transform_edt, binary_closing, binary_opening
from skimage.measure import label, regionprops
import cv2

# ============================================================================
# Flask App Setup
# ============================================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Create demo directories
DEMO_DIR = os.path.join(os.path.dirname(__file__), 'demo_files')
UPLOAD_DIR = os.path.join(DEMO_DIR, 'uploads')
OUTPUT_DIR = os.path.join(DEMO_DIR, 'outputs')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Force CPU-only mode for reliability and consistency
# Change FORCE_CPU to False if you want to try GPU acceleration
FORCE_CPU = True

import onnxruntime as ort
available_providers = ort.get_available_providers()
print(f"Available ONNX providers: {available_providers}")

if FORCE_CPU:
    print("Using CPU mode (FORCE_CPU=True)")
    PROVIDERS = ['CPUExecutionProvider']
elif 'CUDAExecutionProvider' in available_providers:
    print("GPU (CUDA) detected - using GPU acceleration!")
    PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']
else:
    print("Using CPU with optimizations")
    PROVIDERS = ['CPUExecutionProvider']

# Pre-load models
print("Loading background removal models...")
models = {}

# BiRefNet - best overall quality, excellent edge detection
try:
    models['birefnet-general'] = new_session("birefnet-general", providers=PROVIDERS)
    print("  - birefnet-general loaded (BEST quality)")
except Exception as e:
    print(f"  - birefnet-general failed to load: {e}")

# BiRefNet Lite - faster version with good quality
try:
    models['birefnet-general-lite'] = new_session("birefnet-general-lite", providers=PROVIDERS)
    print("  - birefnet-general-lite loaded (Fast + good quality)")
except Exception as e:
    print(f"  - birefnet-general-lite failed to load: {e}")

# isnet-general-use - good for fine details
models['isnet-general-use'] = new_session("isnet-general-use", providers=PROVIDERS)
print("  - isnet-general-use loaded (good for details)")

# u2net - full model, good balance
models['u2net'] = new_session("u2net", providers=PROVIDERS)
print("  - u2net loaded (balanced)")

# u2netp - fastest
models['u2netp'] = new_session("u2netp", providers=PROVIDERS)
print("  - u2netp loaded (fastest)")

# silueta - fast and good for general objects
models['silueta'] = new_session("silueta", providers=PROVIDERS)
print("  - silueta loaded (fast, general objects)")

# u2net_cloth_seg - specialized for clothing
models['u2net_cloth_seg'] = new_session("u2net_cloth_seg", providers=PROVIDERS)
print("  - u2net_cloth_seg loaded (clothing specialized)")

USE_GPU = 'CUDA' in str(PROVIDERS)
print(f"Models loaded! Running on: {'GPU (CUDA)' if USE_GPU else 'CPU'}")

# Pre-warm the preferred model with a tiny test image
print("Warming up models...")
warmup_img = Image.new('RGB', (64, 64), (255, 255, 255))
for name in ['birefnet-general-lite', 'u2netp']:
    if name in models:
        try:
            _ = remove(warmup_img, session=models[name])
            print(f"  - {name} ready")
            break
        except Exception as e:
            print(f"  - {name} warmup failed, skipping")
print("Ready!")

MAX_SIZE = 2048

# ============================================================================
# Fast CPU-based Background Removal (GrabCut, Color Key)
# ============================================================================

def remove_background_grabcut(img, border_percent=5):
    """
    Fast background removal using OpenCV GrabCut.
    Assumes object is centered with background at edges.
    ~0.2-0.5 seconds on CPU.
    """
    img_array = np.array(img)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    else:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    h, w = img_array.shape[:2]

    # Define rectangle - assume object is in center with border as background
    border_x = int(w * border_percent / 100)
    border_y = int(h * border_percent / 100)
    rect = (border_x, border_y, w - 2*border_x, h - 2*border_y)

    # Initialize mask and models
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Run GrabCut
    cv2.grabCut(img_array, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Create binary mask (0,2 = background, 1,3 = foreground)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')

    # Apply some morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)

    # Convert back to RGB and add alpha channel
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    result = np.dstack((img_rgb, mask2))

    return Image.fromarray(result, 'RGBA')


def remove_background_colorkey(img, key_color='green', tolerance=40, spill_removal=True):
    """
    Fast chroma key (green/blue screen) background removal.
    ~0.05-0.1 seconds on CPU - extremely fast!

    key_color: 'green', 'blue', 'white', or RGB tuple like (0, 255, 0)
    tolerance: How much variation from key color to remove (0-100)
    spill_removal: Remove color spill on edges
    """
    img_array = np.array(img.convert('RGB'))

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Define color ranges based on key_color
    if key_color == 'green':
        # Bright green screen range
        lower = np.array([35, 80, 80])
        upper = np.array([85, 255, 255])
    elif key_color == 'blue':
        lower = np.array([100, 80, 80])
        upper = np.array([130, 255, 255])
    elif key_color == 'white':
        # White/light gray background
        lower = np.array([0, 0, 200])
        upper = np.array([180, 30, 255])
    elif isinstance(key_color, tuple):
        # Custom RGB color - convert to HSV range
        color_hsv = cv2.cvtColor(np.uint8([[list(key_color)]]), cv2.COLOR_RGB2HSV)[0][0]
        lower = np.array([max(0, color_hsv[0] - 10), 50, 50])
        upper = np.array([min(180, color_hsv[0] + 10), 255, 255])
    else:
        # Default to green
        lower = np.array([35, 80, 80])
        upper = np.array([85, 255, 255])

    # Adjust tolerance
    tolerance_factor = tolerance / 50  # 50 = default
    hsv_tolerance = int(20 * tolerance_factor)
    lower[0] = max(0, lower[0] - hsv_tolerance)
    upper[0] = min(180, upper[0] + hsv_tolerance)
    lower[1] = max(0, int(lower[1] / tolerance_factor)) if tolerance_factor > 0 else lower[1]

    # Create mask
    mask = cv2.inRange(hsv, lower, upper)

    # Invert mask (we want foreground, not background)
    mask = cv2.bitwise_not(mask)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Optional: Remove green spill from edges
    if spill_removal and key_color == 'green':
        img_array = remove_green_spill(img_array, mask)

    # Create RGBA result
    result = np.dstack((img_array, mask))

    return Image.fromarray(result, 'RGBA')


def remove_green_spill(img_array, mask):
    """Remove green color spill from edges of the subject."""
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


# ============================================================================
# Image Processing Functions (from app.py)
# ============================================================================

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


def clean_edges(img, erode_pixels=2, threshold=128):
    """
    Clean up edges by:
    1. Eroding the mask slightly to remove contaminated edge pixels
    2. Thresholding alpha to remove semi-transparent pixels
    """
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

    # Apply the eroded mask
    data[:, :, 3] = np.where(mask, 255, 0).astype(np.uint8)

    return Image.fromarray(data)


def decontaminate_edges(img, edge_threshold=200):
    """
    Remove color contamination from edge pixels.
    Replaces edge pixel colors with colors from nearby interior pixels.
    """
    if img.mode != 'RGBA':
        return img

    data = np.array(img)
    alpha = data[:, :, 3]

    interior_mask = alpha >= edge_threshold
    edge_mask = (alpha > 0) & (alpha < edge_threshold)

    if not np.any(edge_mask) or not np.any(interior_mask):
        return img

    result = data.copy()
    dist, indices = distance_transform_edt(~interior_mask, return_indices=True)

    edge_y, edge_x = np.where(edge_mask)
    nearest_y = indices[0][edge_y, edge_x]
    nearest_x = indices[1][edge_y, edge_x]

    result[edge_y, edge_x, 0] = data[nearest_y, nearest_x, 0]
    result[edge_y, edge_x, 1] = data[nearest_y, nearest_x, 1]
    result[edge_y, edge_x, 2] = data[nearest_y, nearest_x, 2]

    return Image.fromarray(result)


def soften_edges(img, radius=1.5):
    """Apply gentle edge softening by blurring the alpha channel."""
    if img.mode != 'RGBA':
        return img

    data = np.array(img).astype(float)
    alpha = data[:, :, 3]
    alpha_soft = gaussian_filter(alpha, sigma=radius)
    data[:, :, 3] = alpha_soft
    return Image.fromarray(data.astype(np.uint8))


def apply_alpha_threshold(img, threshold):
    """Apply threshold to alpha channel to adjust edge hardness."""
    if img.mode != 'RGBA':
        return img

    data = np.array(img)
    alpha = data[:, :, 3].astype(float)

    thresh_value = threshold * 2.55

    if threshold < 50:
        factor = (50 - threshold) / 50
        alpha = np.where(alpha > 0, np.clip(alpha + (255 - alpha) * factor * 0.3, 0, 255), alpha)
    else:
        factor = (threshold - 50) / 50
        alpha = np.where(alpha > 128, np.clip(alpha + (255 - alpha) * factor, 0, 255),
                        np.clip(alpha * (1 - factor), 0, 255))

    data[:, :, 3] = alpha.astype(np.uint8)
    return Image.fromarray(data)


def apply_color_correction(img, temperature, tint, brightness=0):
    """
    Apply color temperature, tint, and brightness correction.

    temperature: -100 to +100 (cool to warm)
    tint: -100 to +100 (green to magenta)
    brightness: -100 to +100 (dark to bright)
    """
    if temperature == 0 and tint == 0 and brightness == 0:
        return img

    has_alpha = img.mode == 'RGBA'
    if has_alpha:
        alpha = img.split()[3]
        rgb_img = img.convert('RGB')
    else:
        rgb_img = img.convert('RGB')

    arr = np.array(rgb_img, dtype=np.float32)

    # Brightness
    if brightness != 0:
        brightness_factor = brightness / 100.0
        arr = arr + (brightness_factor * 50)
        arr = np.clip(arr, 0, 255)

    # Temperature
    if temperature != 0:
        temp_factor = temperature / 100.0
        arr[:, :, 0] = np.clip(arr[:, :, 0] + (temp_factor * 30), 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] - (temp_factor * 30), 0, 255)
        if temp_factor > 0:
            arr[:, :, 1] = np.clip(arr[:, :, 1] + (temp_factor * 10), 0, 255)

    # Tint
    if tint != 0:
        tint_factor = tint / 100.0
        arr[:, :, 1] = np.clip(arr[:, :, 1] - (tint_factor * 25), 0, 255)
        if tint_factor > 0:
            arr[:, :, 0] = np.clip(arr[:, :, 0] + (tint_factor * 10), 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] + (tint_factor * 10), 0, 255)

    result = Image.fromarray(arr.astype(np.uint8), 'RGB')

    if has_alpha:
        result = result.convert('RGBA')
        result.putalpha(alpha)

    return result


def apply_edge_smoothing(img, level=2):
    """
    Smooth jagged edges using morphological operations + anti-aliasing.
    level: 1=light, 2=medium, 3=heavy
    """
    if img.mode != 'RGBA':
        return img

    r, g, b, a = img.split()
    alpha_arr = np.array(a, dtype=np.float32)

    binary = (alpha_arr > 128).astype(np.uint8)

    struct = np.array([[0,1,0],
                       [1,1,1],
                       [0,1,0]], dtype=np.uint8)

    iterations = max(2, int(level))

    smoothed = binary_closing(binary, structure=struct, iterations=iterations)
    smoothed = binary_opening(smoothed, structure=struct, iterations=iterations)

    smoothed_alpha = smoothed.astype(np.float32) * 255

    smoothed_pil = Image.fromarray(smoothed_alpha.astype(np.uint8), 'L')
    smoothed_pil = smoothed_pil.filter(ImageFilter.GaussianBlur(radius=0.75))

    alpha_arr_final = np.array(smoothed_pil, dtype=np.float32)
    alpha_arr_final = np.where(alpha_arr_final > 200, 255,
                               np.where(alpha_arr_final < 55, 0, alpha_arr_final))

    smoothed_pil = Image.fromarray(alpha_arr_final.astype(np.uint8), 'L')

    return Image.merge('RGBA', (r, g, b, smoothed_pil))


def apply_drop_shadow(img, level):
    """
    Apply a drop shadow to the image.
    level: 1=soft, 2=medium, 3=strong
    """
    if img.mode != 'RGBA':
        return img

    width, height = img.size

    params = {
        1: {'offset': (4, 4), 'blur': 8, 'opacity': 0.15},
        2: {'offset': (6, 6), 'blur': 12, 'opacity': 0.25},
        3: {'offset': (10, 10), 'blur': 18, 'opacity': 0.35},
    }.get(level, {'offset': (6, 6), 'blur': 12, 'opacity': 0.25})

    offset_x, offset_y = params['offset']
    blur_radius = params['blur']
    opacity = params['opacity']

    alpha = img.split()[3]

    shadow = Image.new('RGBA', (width + offset_x + blur_radius * 2, height + offset_y + blur_radius * 2), (0, 0, 0, 0))
    shadow_alpha = alpha.copy()
    shadow.paste((0, 0, 0, int(255 * opacity)), (blur_radius + offset_x, blur_radius + offset_y), shadow_alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    result = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    shadow_cropped = shadow.crop((blur_radius, blur_radius, blur_radius + width, blur_radius + height))
    result = Image.alpha_composite(result, shadow_cropped)
    result = Image.alpha_composite(result, img)

    return result


def apply_subject_zoom(img, zoom_percent):
    """
    Scale up the subject within the canvas.
    zoom_percent: 100-400 (100 = no zoom, 200 = 2x larger)
    """
    if zoom_percent <= 100 or img.mode != 'RGBA':
        return img

    width, height = img.size
    scale = zoom_percent / 100

    new_width = int(width * scale)
    new_height = int(height * scale)

    scaled = img.resize((new_width, new_height), Image.LANCZOS)

    result = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    paste_x = (width - new_width) // 2
    paste_y = (height - new_height) // 2

    result.paste(scaled, (paste_x, paste_y), scaled)

    return result


def recenter_subject(img):
    """Find the bounding box of the subject and center it."""
    if img.mode != 'RGBA':
        return img

    data = np.array(img)
    alpha = data[:, :, 3]

    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)

    if not np.any(rows) or not np.any(cols):
        return img

    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]
    min_row, max_row = row_indices[0], row_indices[-1]
    min_col, max_col = col_indices[0], col_indices[-1]

    subject_height = max_row - min_row + 1
    subject_width = max_col - min_col + 1

    height, width = alpha.shape

    result = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    subject = img.crop((min_col, min_row, max_col + 1, max_row + 1))

    new_x = (width - subject_width) // 2
    new_y = (height - subject_height) // 2

    result.paste(subject, (new_x, new_y), subject)

    return result


def detect_fragments(img, alpha_threshold=10, min_size=50):
    """Detect connected fragments using alpha channel."""
    if img.mode != 'RGBA':
        return None, {}

    data = np.array(img)
    alpha = data[:, :, 3]

    binary_mask = alpha > alpha_threshold
    labeled = label(binary_mask, connectivity=2)
    regions = regionprops(labeled)

    max_area = 0
    main_label = 0
    for region in regions:
        if region.area > max_area:
            max_area = region.area
            main_label = region.label

    fragment_info = {}
    for region in regions:
        if region.area < min_size:
            continue

        ratio = region.area / max_area if max_area > 0 else 0
        if ratio > 0.5:
            classification = 'main'
        elif ratio > 0.1:
            classification = 'secondary'
        elif ratio > 0.01:
            classification = 'artifact'
        else:
            classification = 'tiny'

        fragment_info[int(region.label)] = {
            'area': int(region.area),
            'centroid': [float(region.centroid[1]), float(region.centroid[0])],
            'bounds': [int(region.bbox[1]), int(region.bbox[0]),
                      int(region.bbox[3]), int(region.bbox[2])],
            'classification': classification,
            'is_main': region.label == main_label
        }

    return labeled, fragment_info


def correct_fisheye(img, strength=0.3):
    """Correct barrel/fisheye distortion from wide-angle lens."""
    if img.mode == 'RGBA':
        img_array = np.array(img)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)
    else:
        img_array = np.array(img.convert('RGB'))
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    h, w = img_cv.shape[:2]

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    k1 = -strength
    k2 = strength * 0.1
    dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float32)

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    corrected = cv2.undistort(img_cv, camera_matrix, dist_coeffs, None, new_camera_matrix)

    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        corrected = corrected[y:y+rh, x:x+rw]
        corrected = cv2.resize(corrected, (w, h), interpolation=cv2.INTER_LANCZOS4)

    if img.mode == 'RGBA':
        corrected = cv2.cvtColor(corrected, cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(corrected, 'RGBA')
    else:
        corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
        return Image.fromarray(corrected, 'RGB')


def composite_on_white(img):
    """Composite RGBA image on white background."""
    if img.mode != 'RGBA':
        return img

    data = np.array(img)
    h, w = data.shape[:2]

    result = np.full((h, w, 3), 255, dtype=np.uint8)
    alpha = data[:, :, 3].astype(float) / 255.0

    for c in range(3):
        result[:, :, c] = (data[:, :, c] * alpha + result[:, :, c] * (1 - alpha)).astype(np.uint8)

    return Image.fromarray(result, 'RGB')


# ============================================================================
# HTML Template for Demo UI
# ============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clothes Mentor BG Removal Test</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #ffffff;
            min-height: 100vh;
            color: #333;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            margin-bottom: 10px;
        }
        .logo { height: 60px; width: auto; }
        h1 {
            text-align: center;
            color: #1a1a2e;
            font-size: 2.2em;
            font-weight: 700;
        }
        .subtitle { text-align: center; color: #666; margin-bottom: 30px; font-weight: 500; }

        /* Upload Section */
        .upload-section {
            background: #f8f9fa;
            border: 2px dashed #ccc;
            border-radius: 16px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s;
        }
        .upload-section:hover { border-color: #0066cc; background: #f0f7ff; }
        .upload-section.dragover { background: #e6f2ff; border-color: #0066cc; }
        .upload-btn {
            background: linear-gradient(135deg, #0066cc, #0088ff);
            border: none;
            padding: 15px 40px;
            border-radius: 30px;
            color: #fff;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 4px 12px rgba(0,102,204,0.3);
        }
        .upload-btn:hover { transform: scale(1.05); box-shadow: 0 6px 16px rgba(0,102,204,0.4); }
        #fileInput { display: none; }

        /* Main Layout */
        .main-layout { display: flex; gap: 30px; flex-wrap: wrap; }

        /* Image Display */
        .image-section {
            flex: 1;
            min-width: 300px;
            background: #f8f9fa;
            border-radius: 16px;
            padding: 20px;
            border: 1px solid #e0e0e0;
        }
        .image-section h3 { margin-bottom: 15px; color: #0066cc; font-weight: 600; }
        .image-container {
            background: repeating-conic-gradient(#e8e8e8 0% 25%, #f5f5f5 0% 50%) 50% / 20px 20px;
            border-radius: 12px;
            min-height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            border: 1px solid #ddd;
        }
        .image-container img {
            max-width: 100%;
            max-height: 500px;
            object-fit: contain;
        }
        .placeholder { color: #999; font-style: italic; }

        /* Controls Panel */
        .controls-section {
            width: 350px;
            background: #f8f9fa;
            border-radius: 16px;
            padding: 20px;
            border: 1px solid #e0e0e0;
        }
        .controls-section h3 { margin-bottom: 20px; color: #0066cc; font-weight: 600; }

        .control-group {
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid #e0e0e0;
        }
        .control-group:last-child { border-bottom: none; }
        .control-group h4 {
            margin-bottom: 12px;
            color: #0066cc;
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .slider-row {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .slider-row label {
            width: 100px;
            font-size: 13px;
            color: #555;
            font-weight: 500;
        }
        .slider-row input[type="range"] {
            flex: 1;
            margin: 0 10px;
            accent-color: #0066cc;
        }
        .slider-row .value {
            width: 40px;
            text-align: right;
            font-size: 13px;
            color: #0066cc;
            font-weight: 600;
        }

        .btn-row { display: flex; gap: 10px; flex-wrap: wrap; }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s;
            font-family: 'Inter', sans-serif;
        }
        .btn-primary {
            background: linear-gradient(135deg, #0066cc, #0088ff);
            color: #fff;
        }
        .btn-secondary {
            background: #fff;
            color: #333;
            border: 1px solid #ccc;
        }
        .btn-secondary:hover { background: #f0f0f0; border-color: #0066cc; }
        .btn-success {
            background: linear-gradient(135deg, #28a745, #20c997);
            color: #fff;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; }

        /* Status/Info */
        .status {
            margin-top: 15px;
            padding: 12px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
        }
        .status.info { background: #e7f3ff; color: #0066cc; border: 1px solid #b3d7ff; }
        .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }

        /* Timings */
        .timings {
            background: #fff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 12px;
            margin-top: 15px;
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
            font-size: 12px;
        }
        .timings div { margin: 4px 0; }
        .timings .label { color: #666; }
        .timings .value { color: #28a745; font-weight: 600; }

        /* Fragment Info */
        .fragment-list {
            max-height: 200px;
            overflow-y: auto;
            background: #fff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
        }
        .fragment-item {
            display: flex;
            justify-content: space-between;
            padding: 8px;
            margin: 4px 0;
            background: #f8f9fa;
            border-radius: 6px;
            font-size: 12px;
        }
        .fragment-item.main { border-left: 3px solid #28a745; }
        .fragment-item.secondary { border-left: 3px solid #0066cc; }
        .fragment-item.artifact { border-left: 3px solid #ffc107; }

        /* Loading */
        .loading {
            display: none;
            align-items: center;
            gap: 10px;
            color: #0066cc;
            margin-bottom: 20px;
            font-weight: 500;
        }
        .loading.active { display: flex; }
        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid #e0e0e0;
            border-top-color: #0066cc;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Download buttons */
        .download-section { margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <img src="/static/logo.png" alt="Logo" class="logo">
            <h1>Clothes Mentor BG Removal Test</h1>
        </div>
        <p class="subtitle">Test & Validate Background Removal and Image Editing Functions</p>

        <!-- Upload Section -->
        <div class="upload-section" id="dropZone">
            <p style="margin-bottom: 20px; color: #666;">Drag & drop an image or click to upload</p>
            <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                Choose Image
            </button>
            <input type="file" id="fileInput" accept="image/*">
        </div>

        <div class="loading" id="loadingIndicator">
            <div class="spinner"></div>
            <span id="loadingText">Processing...</span>
        </div>

        <!-- Main Layout -->
        <div class="main-layout">
            <!-- Original Image -->
            <div class="image-section">
                <h3>Original Image</h3>
                <div class="image-container" id="originalContainer">
                    <span class="placeholder">No image uploaded</span>
                </div>
            </div>

            <!-- Processed Image -->
            <div class="image-section">
                <h3>Processed Result</h3>
                <div class="image-container" id="resultContainer">
                    <span class="placeholder">Upload an image to process</span>
                </div>
                <div class="timings" id="timings" style="display: none;"></div>
            </div>

            <!-- Controls -->
            <div class="controls-section">
                <h3>Image Editing Tools</h3>

                <!-- Background Removal -->
                <div class="control-group">
                    <h4>Background Removal</h4>
                    <div class="slider-row">
                        <label>Model</label>
                        <select id="modelSelect" style="flex: 1; margin: 0 10px; padding: 5px; border-radius: 4px; border: 1px solid #ccc; font-family: 'Inter', sans-serif;">
                            <optgroup label="⚡ Instant (Green Screen)">
                                <option value="colorkey_green" selected>Color Key - Green (Instant)</option>
                                <option value="colorkey_white">Color Key - White (Instant)</option>
                            </optgroup>
                            <optgroup label="🚀 Fast (CPU)">
                                <option value="grabcut">GrabCut (Very Fast)</option>
                                <option value="u2netp">U2NetP (Fast AI)</option>
                                <option value="silueta">Silueta (Fast AI)</option>
                            </optgroup>
                            <optgroup label="👕 Clothing Optimized">
                                <option value="u2net_cloth_seg">U2Net Cloth (Clothing)</option>
                            </optgroup>
                            <optgroup label="🎯 Quality (Slower)">
                                <option value="u2net">U2Net (Balanced)</option>
                                <option value="isnet-general-use">ISNet (Good for straps)</option>
                                <option value="hybrid">Hybrid (Strap + Pattern)</option>
                                <option value="birefnet-general-lite">BiRefNet Lite (Quality)</option>
                                <option value="birefnet-general">BiRefNet (Best quality)</option>
                            </optgroup>
                        </select>
                    </div>
                    <div class="slider-row" id="toleranceRow" style="display: flex;">
                        <label>Tolerance</label>
                        <input type="range" id="colorTolerance" min="10" max="80" value="40">
                        <span class="value" id="colorToleranceVal">40</span>
                    </div>
                    <div class="slider-row">
                        <label>Max Size</label>
                        <select id="maxSizeSelect" style="flex: 1; margin: 0 10px; padding: 5px; border-radius: 4px; border: 1px solid #ccc; font-family: 'Inter', sans-serif;">
                            <option value="512">512px (Fastest)</option>
                            <option value="768">768px (Fast)</option>
                            <option value="1024" selected>1024px (Balanced)</option>
                            <option value="1536">1536px (High quality)</option>
                            <option value="2048">2048px (Best quality)</option>
                        </select>
                    </div>
                    <div class="slider-row">
                        <label>Edge Cleaning</label>
                        <input type="range" id="erodePixels" min="0" max="5" value="1">
                        <span class="value" id="erodePixelsVal">1</span>
                    </div>
                    <p style="font-size: 11px; color: #888; margin: 8px 0;">Tip: Use 0-1 for items with thin straps or fine details</p>
                    <div class="btn-row">
                        <button class="btn btn-primary" id="btnRemoveBg" disabled>Remove Background</button>
                    </div>
                </div>

                <!-- Alpha Threshold -->
                <div class="control-group">
                    <h4>Alpha Threshold</h4>
                    <div class="slider-row">
                        <label>Threshold</label>
                        <input type="range" id="alphaThreshold" min="0" max="100" value="50">
                        <span class="value" id="alphaThresholdVal">50</span>
                    </div>
                    <div class="btn-row">
                        <button class="btn btn-secondary" id="btnApplyThreshold" disabled>Apply</button>
                    </div>
                </div>

                <!-- Color Correction -->
                <div class="control-group">
                    <h4>Color Correction</h4>
                    <div class="slider-row">
                        <label>Temperature</label>
                        <input type="range" id="temperature" min="-100" max="100" value="0">
                        <span class="value" id="temperatureVal">0</span>
                    </div>
                    <div class="slider-row">
                        <label>Tint</label>
                        <input type="range" id="tint" min="-100" max="100" value="0">
                        <span class="value" id="tintVal">0</span>
                    </div>
                    <div class="slider-row">
                        <label>Brightness</label>
                        <input type="range" id="brightness" min="-100" max="100" value="0">
                        <span class="value" id="brightnessVal">0</span>
                    </div>
                    <div class="btn-row">
                        <button class="btn btn-secondary" id="btnApplyColor" disabled>Apply Colors</button>
                        <button class="btn btn-secondary" id="btnResetColor" disabled>Reset</button>
                    </div>
                </div>

                <!-- Edge Smoothing -->
                <div class="control-group">
                    <h4>Edge Smoothing</h4>
                    <div class="slider-row">
                        <label>Level</label>
                        <input type="range" id="edgeSmooth" min="0" max="3" value="0">
                        <span class="value" id="edgeSmoothVal">Off</span>
                    </div>
                    <div class="btn-row">
                        <button class="btn btn-secondary" id="btnApplySmooth" disabled>Apply</button>
                    </div>
                </div>

                <!-- Drop Shadow -->
                <div class="control-group">
                    <h4>Drop Shadow</h4>
                    <div class="slider-row">
                        <label>Intensity</label>
                        <input type="range" id="shadowLevel" min="0" max="3" value="0">
                        <span class="value" id="shadowLevelVal">Off</span>
                    </div>
                    <div class="btn-row">
                        <button class="btn btn-secondary" id="btnApplyShadow" disabled>Apply</button>
                    </div>
                </div>

                <!-- Subject Transform -->
                <div class="control-group">
                    <h4>Subject Transform</h4>
                    <div class="slider-row">
                        <label>Zoom</label>
                        <input type="range" id="subjectZoom" min="100" max="400" value="100">
                        <span class="value" id="subjectZoomVal">100%</span>
                    </div>
                    <div class="btn-row">
                        <button class="btn btn-secondary" id="btnApplyZoom" disabled>Apply Zoom</button>
                        <button class="btn btn-secondary" id="btnRecenter" disabled>Recenter</button>
                    </div>
                </div>

                <!-- Fragment Detection -->
                <div class="control-group">
                    <h4>Fragment Detection</h4>
                    <div class="btn-row">
                        <button class="btn btn-secondary" id="btnDetectFragments" disabled>Detect Fragments</button>
                    </div>
                    <div id="fragmentInfo" style="margin-top: 10px;"></div>
                </div>

                <!-- Additional Tools -->
                <div class="control-group">
                    <h4>Additional Tools</h4>
                    <div class="btn-row">
                        <button class="btn btn-secondary" id="btnDecontaminate" disabled>Decontaminate Edges</button>
                        <button class="btn btn-secondary" id="btnSoftenEdges" disabled>Soften Edges</button>
                    </div>
                    <div class="btn-row" style="margin-top: 10px;">
                        <button class="btn btn-secondary" id="btnFisheye" disabled>Fisheye Correction</button>
                    </div>
                </div>

                <!-- Download -->
                <div class="control-group download-section">
                    <h4>Download</h4>
                    <div class="btn-row">
                        <button class="btn btn-success" id="btnDownloadPng" disabled>Download PNG</button>
                        <button class="btn btn-success" id="btnDownloadWhite" disabled>White Background</button>
                    </div>
                </div>

                <!-- Reset -->
                <div class="control-group">
                    <button class="btn btn-secondary" id="btnReset" disabled style="width: 100%;">Reset All Changes</button>
                </div>

                <div id="statusMessage"></div>
            </div>
        </div>
    </div>

    <script>
        let currentImageId = null;
        let originalImageUrl = null;

        // DOM elements
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const originalContainer = document.getElementById('originalContainer');
        const resultContainer = document.getElementById('resultContainer');
        const loadingIndicator = document.getElementById('loadingIndicator');
        const loadingText = document.getElementById('loadingText');
        const timingsDiv = document.getElementById('timings');
        const statusMessage = document.getElementById('statusMessage');

        // Slider value displays
        const sliders = {
            erodePixels: { el: document.getElementById('erodePixels'), val: document.getElementById('erodePixelsVal') },
            colorTolerance: { el: document.getElementById('colorTolerance'), val: document.getElementById('colorToleranceVal') },
            alphaThreshold: { el: document.getElementById('alphaThreshold'), val: document.getElementById('alphaThresholdVal') },
            temperature: { el: document.getElementById('temperature'), val: document.getElementById('temperatureVal') },
            tint: { el: document.getElementById('tint'), val: document.getElementById('tintVal') },
            brightness: { el: document.getElementById('brightness'), val: document.getElementById('brightnessVal') },
            edgeSmooth: { el: document.getElementById('edgeSmooth'), val: document.getElementById('edgeSmoothVal'), labels: ['Off', 'Light', 'Medium', 'Heavy'] },
            shadowLevel: { el: document.getElementById('shadowLevel'), val: document.getElementById('shadowLevelVal'), labels: ['Off', 'Soft', 'Medium', 'Strong'] },
            subjectZoom: { el: document.getElementById('subjectZoom'), val: document.getElementById('subjectZoomVal'), suffix: '%' }
        };

        // Update slider displays
        Object.entries(sliders).forEach(([key, slider]) => {
            slider.el.addEventListener('input', () => {
                if (slider.labels) {
                    slider.val.textContent = slider.labels[slider.el.value];
                } else if (slider.suffix) {
                    slider.val.textContent = slider.el.value + slider.suffix;
                } else {
                    slider.val.textContent = slider.el.value;
                }
            });
        });

        // Drag and drop
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) handleFile(files[0]);
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
        });

        async function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                showStatus('Please upload an image file', 'error');
                return;
            }

            // Show original
            const reader = new FileReader();
            reader.onload = (e) => {
                originalContainer.innerHTML = `<img src="${e.target.result}" alt="Original">`;
                originalImageUrl = e.target.result;
            };
            reader.readAsDataURL(file);

            // Upload and process
            await uploadAndProcess(file);
        }

        // Show/hide tolerance slider based on model
        document.getElementById('modelSelect').addEventListener('change', function() {
            const toleranceRow = document.getElementById('toleranceRow');
            if (this.value.startsWith('colorkey')) {
                toleranceRow.style.display = 'flex';
            } else {
                toleranceRow.style.display = 'none';
            }
        });

        async function uploadAndProcess(file) {
            const modelSelect = document.getElementById('modelSelect');
            const maxSizeSelect = document.getElementById('maxSizeSelect');
            const modelName = modelSelect.value;
            const tolerance = document.getElementById('colorTolerance').value;
            showLoading(`Removing background with ${modelSelect.options[modelSelect.selectedIndex].text}...`);

            const formData = new FormData();
            formData.append('image', file);
            formData.append('erode_pixels', sliders.erodePixels.el.value);
            formData.append('model', modelName);
            formData.append('max_size', maxSizeSelect.value);
            formData.append('tolerance', tolerance);

            try {
                const response = await fetch('/demo/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                if (data.success) {
                    currentImageId = data.image_id;
                    updateResultImage();
                    showTimings(data.timings);
                    enableControls(true);
                    showStatus('Background removed successfully!', 'success');
                } else {
                    showStatus(data.error || 'Processing failed', 'error');
                }
            } catch (err) {
                showStatus('Error: ' + err.message, 'error');
            }

            hideLoading();
        }

        function updateResultImage() {
            if (!currentImageId) return;
            const timestamp = Date.now();
            resultContainer.innerHTML = `<img src="/demo/image/result/${currentImageId}?t=${timestamp}" alt="Result">`;
        }

        function showTimings(timings) {
            if (!timings) return;
            let html = '<strong>Processing Times:</strong><br>';
            for (const [key, value] of Object.entries(timings)) {
                html += `<div><span class="label">${key}:</span> <span class="value">${value}ms</span></div>`;
            }
            timingsDiv.innerHTML = html;
            timingsDiv.style.display = 'block';
        }

        function showStatus(message, type) {
            statusMessage.className = 'status ' + type;
            statusMessage.textContent = message;
            statusMessage.style.display = 'block';
            setTimeout(() => statusMessage.style.display = 'none', 5000);
        }

        function showLoading(text) {
            loadingText.textContent = text;
            loadingIndicator.classList.add('active');
        }

        function hideLoading() {
            loadingIndicator.classList.remove('active');
        }

        function enableControls(enabled) {
            document.querySelectorAll('.btn').forEach(btn => {
                if (btn.id !== 'btnRemoveBg' || !enabled) {
                    btn.disabled = !enabled;
                }
            });
        }

        // Button handlers
        document.getElementById('btnRemoveBg').addEventListener('click', async () => {
            if (!originalImageUrl) return;
            const response = await fetch(originalImageUrl);
            const blob = await response.blob();
            const file = new File([blob], 'image.png', { type: blob.type });
            await uploadAndProcess(file);
        });

        document.getElementById('btnApplyThreshold').addEventListener('click', async () => {
            showLoading('Applying threshold...');
            try {
                const response = await fetch('/demo/adjust/threshold', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_id: currentImageId,
                        threshold: parseInt(sliders.alphaThreshold.el.value)
                    })
                });
                const data = await response.json();
                if (data.success) {
                    updateResultImage();
                    showStatus('Threshold applied', 'success');
                } else {
                    showStatus(data.error, 'error');
                }
            } catch (err) {
                showStatus('Error: ' + err.message, 'error');
            }
            hideLoading();
        });

        document.getElementById('btnApplyColor').addEventListener('click', async () => {
            showLoading('Applying color correction...');
            try {
                const response = await fetch('/demo/adjust/color', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_id: currentImageId,
                        temperature: parseInt(sliders.temperature.el.value),
                        tint: parseInt(sliders.tint.el.value),
                        brightness: parseInt(sliders.brightness.el.value)
                    })
                });
                const data = await response.json();
                if (data.success) {
                    updateResultImage();
                    showStatus('Color correction applied', 'success');
                } else {
                    showStatus(data.error, 'error');
                }
            } catch (err) {
                showStatus('Error: ' + err.message, 'error');
            }
            hideLoading();
        });

        document.getElementById('btnResetColor').addEventListener('click', () => {
            sliders.temperature.el.value = 0;
            sliders.tint.el.value = 0;
            sliders.brightness.el.value = 0;
            sliders.temperature.val.textContent = '0';
            sliders.tint.val.textContent = '0';
            sliders.brightness.val.textContent = '0';
        });

        document.getElementById('btnApplySmooth').addEventListener('click', async () => {
            showLoading('Applying edge smoothing...');
            try {
                const response = await fetch('/demo/adjust/smooth', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_id: currentImageId,
                        level: parseInt(sliders.edgeSmooth.el.value)
                    })
                });
                const data = await response.json();
                if (data.success) {
                    updateResultImage();
                    showStatus('Edge smoothing applied', 'success');
                } else {
                    showStatus(data.error, 'error');
                }
            } catch (err) {
                showStatus('Error: ' + err.message, 'error');
            }
            hideLoading();
        });

        document.getElementById('btnApplyShadow').addEventListener('click', async () => {
            showLoading('Applying drop shadow...');
            try {
                const response = await fetch('/demo/adjust/shadow', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_id: currentImageId,
                        level: parseInt(sliders.shadowLevel.el.value)
                    })
                });
                const data = await response.json();
                if (data.success) {
                    updateResultImage();
                    showStatus('Drop shadow applied', 'success');
                } else {
                    showStatus(data.error, 'error');
                }
            } catch (err) {
                showStatus('Error: ' + err.message, 'error');
            }
            hideLoading();
        });

        document.getElementById('btnApplyZoom').addEventListener('click', async () => {
            showLoading('Applying zoom...');
            try {
                const response = await fetch('/demo/adjust/zoom', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_id: currentImageId,
                        zoom: parseInt(sliders.subjectZoom.el.value)
                    })
                });
                const data = await response.json();
                if (data.success) {
                    updateResultImage();
                    showStatus('Zoom applied', 'success');
                } else {
                    showStatus(data.error, 'error');
                }
            } catch (err) {
                showStatus('Error: ' + err.message, 'error');
            }
            hideLoading();
        });

        document.getElementById('btnRecenter').addEventListener('click', async () => {
            showLoading('Recentering subject...');
            try {
                const response = await fetch('/demo/adjust/recenter', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_id: currentImageId })
                });
                const data = await response.json();
                if (data.success) {
                    updateResultImage();
                    showStatus('Subject recentered', 'success');
                } else {
                    showStatus(data.error, 'error');
                }
            } catch (err) {
                showStatus('Error: ' + err.message, 'error');
            }
            hideLoading();
        });

        document.getElementById('btnDetectFragments').addEventListener('click', async () => {
            showLoading('Detecting fragments...');
            try {
                const response = await fetch('/demo/fragments/detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_id: currentImageId })
                });
                const data = await response.json();
                if (data.success) {
                    displayFragments(data.fragments);
                    showStatus(`Found ${data.fragment_count} fragments`, 'success');
                } else {
                    showStatus(data.error, 'error');
                }
            } catch (err) {
                showStatus('Error: ' + err.message, 'error');
            }
            hideLoading();
        });

        function displayFragments(fragments) {
            const container = document.getElementById('fragmentInfo');
            if (!fragments || Object.keys(fragments).length === 0) {
                container.innerHTML = '<p style="color: #666; font-size: 12px;">No fragments detected</p>';
                return;
            }

            let html = '<div class="fragment-list">';
            for (const [id, info] of Object.entries(fragments)) {
                html += `<div class="fragment-item ${info.classification}">
                    <span>#${id} - ${info.classification}</span>
                    <span>${info.area.toLocaleString()} px</span>
                </div>`;
            }
            html += '</div>';
            container.innerHTML = html;
        }

        document.getElementById('btnDecontaminate').addEventListener('click', async () => {
            showLoading('Decontaminating edges...');
            try {
                const response = await fetch('/demo/adjust/decontaminate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_id: currentImageId })
                });
                const data = await response.json();
                if (data.success) {
                    updateResultImage();
                    showStatus('Edges decontaminated', 'success');
                } else {
                    showStatus(data.error, 'error');
                }
            } catch (err) {
                showStatus('Error: ' + err.message, 'error');
            }
            hideLoading();
        });

        document.getElementById('btnSoftenEdges').addEventListener('click', async () => {
            showLoading('Softening edges...');
            try {
                const response = await fetch('/demo/adjust/soften', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_id: currentImageId })
                });
                const data = await response.json();
                if (data.success) {
                    updateResultImage();
                    showStatus('Edges softened', 'success');
                } else {
                    showStatus(data.error, 'error');
                }
            } catch (err) {
                showStatus('Error: ' + err.message, 'error');
            }
            hideLoading();
        });

        document.getElementById('btnFisheye').addEventListener('click', async () => {
            showLoading('Applying fisheye correction...');
            try {
                const response = await fetch('/demo/adjust/fisheye', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_id: currentImageId })
                });
                const data = await response.json();
                if (data.success) {
                    updateResultImage();
                    showStatus('Fisheye correction applied', 'success');
                } else {
                    showStatus(data.error, 'error');
                }
            } catch (err) {
                showStatus('Error: ' + err.message, 'error');
            }
            hideLoading();
        });

        document.getElementById('btnDownloadPng').addEventListener('click', () => {
            if (currentImageId) {
                window.location.href = `/demo/download/${currentImageId}?bg=transparent`;
            }
        });

        document.getElementById('btnDownloadWhite').addEventListener('click', () => {
            if (currentImageId) {
                window.location.href = `/demo/download/${currentImageId}?bg=white`;
            }
        });

        document.getElementById('btnReset').addEventListener('click', async () => {
            showLoading('Resetting...');
            try {
                const response = await fetch('/demo/reset', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_id: currentImageId })
                });
                const data = await response.json();
                if (data.success) {
                    updateResultImage();
                    // Reset sliders
                    sliders.alphaThreshold.el.value = 50;
                    sliders.alphaThreshold.val.textContent = '50';
                    sliders.temperature.el.value = 0;
                    sliders.temperature.val.textContent = '0';
                    sliders.tint.el.value = 0;
                    sliders.tint.val.textContent = '0';
                    sliders.brightness.el.value = 0;
                    sliders.brightness.val.textContent = '0';
                    sliders.edgeSmooth.el.value = 0;
                    sliders.edgeSmooth.val.textContent = 'Off';
                    sliders.shadowLevel.el.value = 0;
                    sliders.shadowLevel.val.textContent = 'Off';
                    sliders.subjectZoom.el.value = 100;
                    sliders.subjectZoom.val.textContent = '100%';
                    showStatus('Reset to original processed image', 'success');
                } else {
                    showStatus(data.error, 'error');
                }
            } catch (err) {
                showStatus('Error: ' + err.message, 'error');
            }
            hideLoading();
        });

        // Enable remove bg button when original is loaded
        document.getElementById('btnRemoveBg').disabled = false;
    </script>
</body>
</html>
'''

# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/demo/upload', methods=['POST'])
def demo_upload():
    """Upload and process an image with background removal."""
    timings = {}
    total_start = time.time()

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    erode_pixels = int(request.form.get('erode_pixels', 1))  # Default to 1 for better detail preservation
    model_name = request.form.get('model', 'colorkey_green')  # Green screen by default
    max_size = int(request.form.get('max_size', 2048))  # Configurable resolution
    tolerance = int(request.form.get('tolerance', 40))  # Color key tolerance

    try:
        image_id = str(uuid.uuid4())

        # Read image
        step_start = time.time()
        img = Image.open(file.stream)
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        timings['read'] = round((time.time() - step_start) * 1000)

        # Resize if needed
        step_start = time.time()
        img, was_resized = resize_image(img, max_size=max_size)
        timings['resize'] = round((time.time() - step_start) * 1000)
        timings['max_size'] = max_size

        # Save original
        original_path = os.path.join(UPLOAD_DIR, f'{image_id}_original.png')
        img.save(original_path, 'PNG')

        # Remove background using selected model
        step_start = time.time()

        if model_name == 'colorkey_green':
            # Instant green screen removal
            output = remove_background_colorkey(img, key_color='green', tolerance=tolerance)
            timings['model'] = 'colorkey_green'

        elif model_name == 'colorkey_white':
            # Instant white background removal
            output = remove_background_colorkey(img, key_color='white', tolerance=tolerance)
            timings['model'] = 'colorkey_white'

        elif model_name == 'grabcut':
            # Fast GrabCut
            output = remove_background_grabcut(img)
            timings['model'] = 'grabcut'

        elif model_name == 'hybrid':
            # Hybrid mode: combine ISNet (good for straps) with U2NetP (good for patterns)
            isnet_session = models.get('isnet-general-use')
            u2net_session = models.get('u2netp')

            output_isnet = remove(img, session=isnet_session)
            output_u2net = remove(img, session=u2net_session)

            alpha_isnet = np.array(output_isnet)[:, :, 3]
            alpha_u2net = np.array(output_u2net)[:, :, 3]

            combined_alpha = np.maximum(alpha_isnet, alpha_u2net)

            output_data = np.array(output_u2net)
            output_data[:, :, 3] = combined_alpha
            output = Image.fromarray(output_data)
            timings['model'] = 'hybrid (isnet + u2netp)'

        else:
            # AI model via rembg
            session = models.get(model_name, models.get('u2netp'))
            if session is None:
                return jsonify({'error': f'Model {model_name} not available'}), 400
            output = remove(img, session=session)
            timings['model'] = model_name

        timings['remove_bg'] = round((time.time() - step_start) * 1000)

        # Clean edges (less aggressive by default)
        step_start = time.time()
        output = clean_edges(output, erode_pixels=erode_pixels)
        timings['clean_edges'] = round((time.time() - step_start) * 1000)

        # Save output
        step_start = time.time()
        output_path = os.path.join(OUTPUT_DIR, f'{image_id}_output.png')
        output.save(output_path, 'PNG')
        timings['save'] = round((time.time() - step_start) * 1000)

        timings['total'] = round((time.time() - total_start) * 1000)

        return jsonify({
            'success': True,
            'image_id': image_id,
            'timings': timings,
            'original_size': list(img.size),
            'was_resized': was_resized
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/demo/image/<image_type>/<image_id>')
def demo_get_image(image_type, image_id):
    """Serve processed images."""
    if image_type == 'original':
        path = os.path.join(UPLOAD_DIR, f'{image_id}_original.png')
    elif image_type == 'result':
        # Check for adjusted version first
        adjusted_path = os.path.join(OUTPUT_DIR, f'{image_id}_adjusted.png')
        output_path = os.path.join(OUTPUT_DIR, f'{image_id}_output.png')
        path = adjusted_path if os.path.exists(adjusted_path) else output_path
    else:
        return jsonify({'error': 'Invalid image type'}), 400

    if not os.path.exists(path):
        return jsonify({'error': 'Image not found'}), 404

    return send_file(path, mimetype='image/png')


@app.route('/demo/adjust/threshold', methods=['POST'])
def demo_adjust_threshold():
    """Apply alpha threshold adjustment."""
    data = request.json
    image_id = data.get('image_id')
    threshold = data.get('threshold', 50)

    try:
        img = load_current_image(image_id)
        if img is None:
            return jsonify({'error': 'Image not found'}), 404

        img = apply_alpha_threshold(img, threshold)
        save_adjusted_image(image_id, img)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/demo/adjust/color', methods=['POST'])
def demo_adjust_color():
    """Apply color correction."""
    data = request.json
    image_id = data.get('image_id')
    temperature = data.get('temperature', 0)
    tint = data.get('tint', 0)
    brightness = data.get('brightness', 0)

    try:
        img = load_current_image(image_id)
        if img is None:
            return jsonify({'error': 'Image not found'}), 404

        img = apply_color_correction(img, temperature, tint, brightness)
        save_adjusted_image(image_id, img)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/demo/adjust/smooth', methods=['POST'])
def demo_adjust_smooth():
    """Apply edge smoothing."""
    data = request.json
    image_id = data.get('image_id')
    level = data.get('level', 2)

    try:
        img = load_current_image(image_id)
        if img is None:
            return jsonify({'error': 'Image not found'}), 404

        if level > 0:
            img = apply_edge_smoothing(img, level)
        save_adjusted_image(image_id, img)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/demo/adjust/shadow', methods=['POST'])
def demo_adjust_shadow():
    """Apply drop shadow."""
    data = request.json
    image_id = data.get('image_id')
    level = data.get('level', 2)

    try:
        img = load_current_image(image_id)
        if img is None:
            return jsonify({'error': 'Image not found'}), 404

        if level > 0:
            img = apply_drop_shadow(img, level)
        save_adjusted_image(image_id, img)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/demo/adjust/zoom', methods=['POST'])
def demo_adjust_zoom():
    """Apply subject zoom."""
    data = request.json
    image_id = data.get('image_id')
    zoom = data.get('zoom', 100)

    try:
        img = load_current_image(image_id)
        if img is None:
            return jsonify({'error': 'Image not found'}), 404

        img = apply_subject_zoom(img, zoom)
        save_adjusted_image(image_id, img)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/demo/adjust/recenter', methods=['POST'])
def demo_adjust_recenter():
    """Recenter the subject."""
    data = request.json
    image_id = data.get('image_id')

    try:
        img = load_current_image(image_id)
        if img is None:
            return jsonify({'error': 'Image not found'}), 404

        img = recenter_subject(img)
        save_adjusted_image(image_id, img)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/demo/adjust/decontaminate', methods=['POST'])
def demo_adjust_decontaminate():
    """Decontaminate edges."""
    data = request.json
    image_id = data.get('image_id')

    try:
        img = load_current_image(image_id)
        if img is None:
            return jsonify({'error': 'Image not found'}), 404

        img = decontaminate_edges(img)
        save_adjusted_image(image_id, img)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/demo/adjust/soften', methods=['POST'])
def demo_adjust_soften():
    """Soften edges."""
    data = request.json
    image_id = data.get('image_id')

    try:
        img = load_current_image(image_id)
        if img is None:
            return jsonify({'error': 'Image not found'}), 404

        img = soften_edges(img, radius=1.5)
        save_adjusted_image(image_id, img)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/demo/adjust/fisheye', methods=['POST'])
def demo_adjust_fisheye():
    """Apply fisheye correction."""
    data = request.json
    image_id = data.get('image_id')

    try:
        # Load original for fisheye correction
        original_path = os.path.join(UPLOAD_DIR, f'{image_id}_original.png')
        if not os.path.exists(original_path):
            return jsonify({'error': 'Original image not found'}), 404

        img = Image.open(original_path)
        img = correct_fisheye(img, strength=0.25)

        # Re-process with background removal
        if img.mode != 'RGB':
            img = img.convert('RGB')

        output = remove(img, session=session)
        output = clean_edges(output, erode_pixels=3)

        # Save as new output (overwrite)
        output_path = os.path.join(OUTPUT_DIR, f'{image_id}_output.png')
        output.save(output_path, 'PNG')

        # Remove adjusted if exists
        adjusted_path = os.path.join(OUTPUT_DIR, f'{image_id}_adjusted.png')
        if os.path.exists(adjusted_path):
            os.remove(adjusted_path)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/demo/fragments/detect', methods=['POST'])
def demo_detect_fragments():
    """Detect fragments in the image."""
    data = request.json
    image_id = data.get('image_id')

    try:
        img = load_current_image(image_id)
        if img is None:
            return jsonify({'error': 'Image not found'}), 404

        labeled, fragment_info = detect_fragments(img)

        if labeled is None:
            return jsonify({'error': 'Image must have alpha channel'}), 400

        return jsonify({
            'success': True,
            'fragment_count': len(fragment_info),
            'fragments': fragment_info
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/demo/download/<image_id>')
def demo_download(image_id):
    """Download the processed image."""
    bg = request.args.get('bg', 'transparent')

    img = load_current_image(image_id)
    if img is None:
        return jsonify({'error': 'Image not found'}), 404

    if bg == 'white':
        img = composite_on_white(img)

    buffer = BytesIO()
    img.save(buffer, 'PNG')
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype='image/png',
        as_attachment=True,
        download_name=f'demo_{image_id}.png'
    )


@app.route('/demo/reset', methods=['POST'])
def demo_reset():
    """Reset to original processed image."""
    data = request.json
    image_id = data.get('image_id')

    try:
        adjusted_path = os.path.join(OUTPUT_DIR, f'{image_id}_adjusted.png')
        if os.path.exists(adjusted_path):
            os.remove(adjusted_path)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Helper Functions
# ============================================================================

def load_current_image(image_id):
    """Load the current state of an image (adjusted or output)."""
    adjusted_path = os.path.join(OUTPUT_DIR, f'{image_id}_adjusted.png')
    output_path = os.path.join(OUTPUT_DIR, f'{image_id}_output.png')

    if os.path.exists(adjusted_path):
        return Image.open(adjusted_path)
    elif os.path.exists(output_path):
        return Image.open(output_path)
    return None


def save_adjusted_image(image_id, img):
    """Save the adjusted version of an image."""
    adjusted_path = os.path.join(OUTPUT_DIR, f'{image_id}_adjusted.png')
    img.save(adjusted_path, 'PNG')


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("KaizenStudio Demo - Background Removal & Image Editing")
    print("="*60)
    print(f"\nDemo files directory: {DEMO_DIR}")
    print("\nOpen http://localhost:5001 in your browser")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
