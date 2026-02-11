"""
Standalone Background Removal Model Benchmark
==============================================
Tests all available rembg models + hybrid approaches (AI + CV colorkey,
AI + green spill removal, pure greenscreen) against real product images.

Goal: Find the best background removal quality (zero green remnants)
under 2 seconds on CPU, using any combination of techniques.

Usage:
    venv/Scripts/python.exe test_bg_removal.py
    venv/Scripts/python.exe test_bg_removal.py --image uploads/some_image.png
    venv/Scripts/python.exe test_bg_removal.py --all-images
    venv/Scripts/python.exe test_bg_removal.py --models u2netp silueta

Results saved to test_results/ folder.
"""

import os
import sys
import time
import argparse
import glob
from pathlib import Path

# Force CPU-only ONNX Runtime (suppress TensorRT/CUDA warnings)
os.environ["ORT_TENSORRT_UNAVAILABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import onnxruntime
onnxruntime.set_default_logger_severity(3)  # ERROR only
from rembg import remove, new_session

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = [
    "u2netp",               # Current default - fast
    "u2net",                # Full quality U2Net
    "silueta",              # Fastest
    "isnet-general-use",    # ISNet general
    "birefnet-general-lite",# BiRefNet lite
    "birefnet-general",     # BiRefNet SOTA
]

# bria-rmbg requires accepting a license on HuggingFace, skip by default
# Add it here if you have access:
# MODELS.append("bria-rmbg")

TIME_LIMIT = 2.0  # seconds
RESULTS_DIR = Path("test_results")
UPLOADS_DIR = Path("uploads")
ALPHA_THRESHOLD = 128  # Match app.py behavior

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_image():
    """Find the most recent *_original.png in uploads/."""
    originals = list(UPLOADS_DIR.glob("*_original.png"))
    if not originals:
        print(f"ERROR: No *_original.png files found in {UPLOADS_DIR}/")
        print("Upload an image through the app first, or specify --image path.")
        sys.exit(1)
    originals.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return originals[0]


def apply_alpha_threshold(rgba_img, threshold=ALPHA_THRESHOLD):
    """Threshold alpha channel: below -> 0, above -> 255. Matches app.py."""
    arr = np.array(rgba_img)
    alpha = arr[:, :, 3]
    alpha = np.where(alpha < threshold, 0, 255).astype(np.uint8)
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, "RGBA")


def composite_on_white(rgba_img):
    """Paste RGBA image onto a white background."""
    white = Image.new("RGB", rgba_img.size, (255, 255, 255))
    white.paste(rgba_img, mask=rgba_img.split()[3])
    return white


def analyze_green_remnants(rgba_img):
    """
    Check for green-ish pixels remaining in the opaque region.
    Returns (count, percentage) of pixels that are suspiciously green.
    """
    arr = np.array(rgba_img)
    alpha = arr[:, :, 3]
    opaque_mask = alpha > 0

    r = arr[:, :, 0].astype(float)
    g = arr[:, :, 1].astype(float)
    b = arr[:, :, 2].astype(float)

    # A pixel is "green remnant" if green dominates red and blue
    # and it sits near the edge (semi-transparent or recently thresholded)
    green_dominant = (g > r + 30) & (g > b + 30) & (g > 80)
    green_and_opaque = green_dominant & opaque_mask

    total_opaque = int(np.sum(opaque_mask))
    green_count = int(np.sum(green_and_opaque))
    pct = (green_count / total_opaque * 100) if total_opaque > 0 else 0.0
    return green_count, pct


# ---------------------------------------------------------------------------
# Hybrid removal techniques
# ---------------------------------------------------------------------------

def greenscreen_colorkey(img_rgb):
    """
    Pure HSV chroma-key green screen removal. Very fast (~0.05s).
    Multi-pass to catch bright green, shadowed green, and dark green.
    """
    img_array = np.array(img_rgb)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Pass 1: Bright green
    mask1 = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([85, 255, 255]))
    # Pass 2: Shadowed green
    mask2 = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([90, 255, 180]))
    # Pass 3: Very dark green shadows
    mask3 = cv2.inRange(hsv, np.array([25, 15, 15]), np.array([95, 200, 120]))

    mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)
    mask = cv2.bitwise_not(mask)  # Invert: foreground = white

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Anti-alias edges
    mask_f = mask.astype(np.float32) / 255.0
    mask_f = cv2.GaussianBlur(mask_f, (3, 3), 0)
    mask_f = np.clip((mask_f - 0.2) * 1.5, 0, 1)
    mask = (mask_f * 255).astype(np.uint8)

    result = np.dstack((img_array, mask))
    return Image.fromarray(result, "RGBA")


def remove_green_spill_cv(img_array, mask):
    """Desaturate green in edge regions to remove color spill."""
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=2)
    eroded = cv2.erode(mask, kernel, iterations=2)
    edge_mask = dilated - eroded

    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    green_edge = (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85) & (edge_mask > 0)
    hsv[green_edge, 1] = (hsv[green_edge, 1] * 0.3).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def hybrid_ai_then_despill(img_rgb, session):
    """
    AI background removal followed by green spill removal on edges.
    Combines neural net accuracy with CV-based green cleanup.
    """
    # Step 1: AI removal
    raw = remove(img_rgb, session=session)
    arr = np.array(raw)

    # Step 2: Threshold alpha
    alpha = arr[:, :, 3]
    alpha = np.where(alpha < ALPHA_THRESHOLD, 0, 255).astype(np.uint8)

    # Step 3: Green spill removal on the foreground edges
    rgb = arr[:, :, :3]
    rgb = remove_green_spill_cv(rgb, alpha)

    arr[:, :, :3] = rgb
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, "RGBA")


def hybrid_ai_plus_colorkey(img_rgb, session):
    """
    Intersect AI mask with colorkey mask for tighter boundaries.
    AI provides subject detection; colorkey ensures no green survives.
    """
    # Step 1: AI removal
    raw_ai = remove(img_rgb, session=session)
    ai_arr = np.array(raw_ai)
    ai_alpha = ai_arr[:, :, 3]
    ai_alpha = np.where(ai_alpha < ALPHA_THRESHOLD, 0, 255).astype(np.uint8)

    # Step 2: Colorkey mask (what is NOT green)
    img_array = np.array(img_rgb)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Detect green
    mask1 = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([85, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([90, 255, 180]))
    mask3 = cv2.inRange(hsv, np.array([25, 15, 15]), np.array([95, 200, 120]))
    green_mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)

    # Foreground = not green
    colorkey_fg = cv2.bitwise_not(green_mask)

    # Slight dilate on colorkey to avoid eating into subject
    kernel = np.ones((3, 3), np.uint8)
    colorkey_fg = cv2.dilate(colorkey_fg, kernel, iterations=1)

    # Step 3: Intersect - pixel must pass BOTH AI and colorkey
    combined_alpha = cv2.bitwise_and(ai_alpha, colorkey_fg)

    # Step 4: Green spill removal
    rgb = remove_green_spill_cv(img_array, combined_alpha)

    result = np.dstack((rgb, combined_alpha))
    return Image.fromarray(result, "RGBA")


def hybrid_colorkey_then_ai_refine(img_rgb, session):
    """
    Colorkey first (fast, catches obvious green), then AI to fix
    areas where colorkey was too aggressive or missed non-green BG.
    Uses AI mask as the primary, but zeroes out any pixel that the
    colorkey identified as green.
    """
    # Step 1: AI removal (primary mask)
    raw_ai = remove(img_rgb, session=session)
    ai_arr = np.array(raw_ai)
    ai_alpha = ai_arr[:, :, 3]
    ai_alpha = np.where(ai_alpha < ALPHA_THRESHOLD, 0, 255).astype(np.uint8)

    # Step 2: Strict green detection (only obvious green, no dilation)
    img_array = np.array(img_rgb)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    green_strict = cv2.inRange(hsv, np.array([35, 60, 60]), np.array([85, 255, 255]))

    # Erode green mask slightly so we don't bite into subject
    kernel = np.ones((2, 2), np.uint8)
    green_strict = cv2.erode(green_strict, kernel, iterations=1)

    # Step 3: Any pixel that is definitely green -> force transparent
    combined = ai_alpha.copy()
    combined[green_strict > 0] = 0

    # Step 4: Despill
    rgb = remove_green_spill_cv(img_array, combined)

    result = np.dstack((rgb, combined))
    return Image.fromarray(result, "RGBA")


def hybrid_ai_grabcut(img_rgb, session):
    """
    AI mask used as initialization for GrabCut refinement.
    GrabCut uses color distribution modeling (GMM) to refine the boundary,
    which can fix edges where AI was uncertain.
    """
    # Step 1: AI removal to get initial mask
    raw_ai = remove(img_rgb, session=session)
    ai_arr = np.array(raw_ai)
    ai_alpha = ai_arr[:, :, 3]
    ai_alpha_bin = np.where(ai_alpha < ALPHA_THRESHOLD, 0, 255).astype(np.uint8)

    # Step 2: Convert AI mask to GrabCut format
    img_array = np.array(img_rgb)
    gc_mask = np.where(ai_alpha_bin > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)

    # Definite foreground: erode the AI mask to find confident interior
    kernel = np.ones((15, 15), np.uint8)
    definite_fg = cv2.erode(ai_alpha_bin, kernel, iterations=1)
    gc_mask[definite_fg > 0] = cv2.GC_FGD

    # Definite background: far from any foreground
    definite_bg = cv2.dilate(ai_alpha_bin, kernel, iterations=2)
    gc_mask[definite_bg == 0] = cv2.GC_BGD

    # Step 3: Run GrabCut (limited iterations for speed)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_array, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        # GrabCut can fail on some images; fall back to AI-only
        gc_mask = np.where(ai_alpha_bin > 0, cv2.GC_FGD, cv2.GC_BGD).astype(np.uint8)

    # Step 4: Extract final mask
    final_mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # Step 5: Remove green spill
    rgb = remove_green_spill_cv(img_array, final_mask)

    result = np.dstack((rgb, final_mask))
    return Image.fromarray(result, "RGBA")


def hybrid_kmeans_colorkey(img_rgb, n_clusters=5):
    """
    K-Means color clustering to segment the image, then identify
    the green cluster(s) as background. Works well for uniform
    green screens where the green is a dominant color cluster.
    """
    img_array = np.array(img_rgb)
    h, w = img_array.shape[:2]

    # Reshape for K-Means
    pixels = img_array.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    centers = centers.astype(np.uint8)
    labels = labels.reshape(h, w)

    # Identify green clusters (center color is green-ish)
    green_clusters = []
    for i, c in enumerate(centers):
        r, g, b = int(c[0]), int(c[1]), int(c[2])
        # Check if this cluster center is green-dominant
        if g > r + 20 and g > b + 20 and g > 60:
            green_clusters.append(i)

    # Build mask: foreground = not a green cluster
    mask = np.ones((h, w), dtype=np.uint8) * 255
    for gc in green_clusters:
        mask[labels == gc] = 0

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Anti-alias
    mask_f = mask.astype(np.float32) / 255.0
    mask_f = cv2.GaussianBlur(mask_f, (3, 3), 0)
    mask_f = np.clip((mask_f - 0.15) * 1.3, 0, 1)
    mask = (mask_f * 255).astype(np.uint8)

    # Despill
    rgb = remove_green_spill_cv(img_array, mask)

    result = np.dstack((rgb, mask))
    return Image.fromarray(result, "RGBA")


def hybrid_contour_cleanup(rgba_img):
    """
    Post-process an RGBA result by finding contours, keeping only the
    largest one (the product), and removing small disconnected fragments.
    """
    arr = np.array(rgba_img)
    alpha = arr[:, :, 3]

    # Find contours
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return rgba_img

    # Keep only the largest contour (the product)
    largest = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(alpha)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)

    # Also keep any contour larger than 1% of the largest
    largest_area = cv2.contourArea(largest)
    for c in contours:
        if cv2.contourArea(c) > largest_area * 0.01:
            cv2.drawContours(clean_mask, [c], -1, 255, -1)

    arr[:, :, 3] = cv2.bitwise_and(alpha, clean_mask)
    return Image.fromarray(arr, "RGBA")


# ---------------------------------------------------------------------------
# Visual output helpers
# ---------------------------------------------------------------------------

def make_label(text, width, height=30):
    """Create a simple text label image."""
    label = Image.new("RGB", (width, height), (40, 40, 40))
    draw = ImageDraw.Draw(label)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (width - tw) // 2
    y = (height - th) // 2
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return label


def build_comparison_strip(original, results, thumb_width=400):
    """
    Build a horizontal comparison strip:
    Original | Model1 result | Model2 result | ...
    Each with a label below showing model name, time, and pass/fail.
    """
    # Resize all to same width, preserving aspect ratio
    aspect = original.height / original.width
    thumb_h = int(thumb_width * aspect)

    panels = []

    # Original panel
    orig_thumb = original.resize((thumb_width, thumb_h), Image.LANCZOS)
    orig_on_white = composite_on_white(orig_thumb.convert("RGBA"))
    orig_label = make_label("ORIGINAL", thumb_width)
    panel = Image.new("RGB", (thumb_width, thumb_h + 30), (40, 40, 40))
    panel.paste(orig_on_white, (0, 0))
    panel.paste(orig_label, (0, thumb_h))
    panels.append(panel)

    # Result panels
    for r in results:
        result_rgba = r["image"]
        result_thumb = result_rgba.resize((thumb_width, thumb_h), Image.LANCZOS)
        on_white = composite_on_white(result_thumb)

        status = "PASS" if r["time"] < TIME_LIMIT else "SLOW"
        label_text = f'{r["model"]}  {r["time"]:.2f}s  [{status}]'
        label = make_label(label_text, thumb_width)

        panel = Image.new("RGB", (thumb_width, thumb_h + 30), (40, 40, 40))
        panel.paste(on_white, (0, 0))
        panel.paste(label, (0, thumb_h))
        panels.append(panel)

    # Stitch horizontally
    total_w = thumb_width * len(panels) + 4 * (len(panels) - 1)
    strip = Image.new("RGB", (total_w, thumb_h + 30), (30, 30, 30))
    x = 0
    for p in panels:
        strip.paste(p, (x, 0))
        x += thumb_width + 4

    return strip


def build_detail_crop(original, results, thumb_width=500):
    """
    Build a detail view: crops the center-bottom 40% of each result
    (where straps/loops/edges typically are) and composites on white
    to reveal any green remnants.
    """
    aspect = original.height / original.width
    thumb_h = int(thumb_width * aspect)

    # Crop region: bottom 40%, full width
    crop_top = int(thumb_h * 0.5)
    crop_h = thumb_h - crop_top

    panels = []

    # Original detail
    orig_thumb = original.resize((thumb_width, thumb_h), Image.LANCZOS).convert("RGBA")
    orig_crop = composite_on_white(orig_thumb).crop((0, crop_top, thumb_width, thumb_h))
    label = make_label("ORIGINAL (detail)", thumb_width)
    panel = Image.new("RGB", (thumb_width, crop_h + 30), (40, 40, 40))
    panel.paste(orig_crop, (0, 0))
    panel.paste(label, (0, crop_h))
    panels.append(panel)

    for r in results:
        result_thumb = r["image"].resize((thumb_width, thumb_h), Image.LANCZOS)
        detail = composite_on_white(result_thumb).crop((0, crop_top, thumb_width, thumb_h))

        green_count, green_pct = r["green_count"], r["green_pct"]
        label_text = f'{r["model"]}  green:{green_count}px ({green_pct:.2f}%)'
        label = make_label(label_text, thumb_width)

        panel = Image.new("RGB", (thumb_width, crop_h + 30), (40, 40, 40))
        panel.paste(detail, (0, 0))
        panel.paste(label, (0, crop_h))
        panels.append(panel)

    total_w = thumb_width * len(panels) + 4 * (len(panels) - 1)
    strip = Image.new("RGB", (total_w, crop_h + 30), (30, 30, 30))
    x = 0
    for p in panels:
        strip.paste(p, (x, 0))
        x += thumb_width + 4

    return strip


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def benchmark_image(image_path, models=MODELS):
    """Run all models + hybrid approaches against a single image."""
    print(f"\n{'='*70}")
    print(f"  Image: {image_path}")
    print(f"{'='*70}")

    img = Image.open(image_path).convert("RGB")
    print(f"  Size: {img.width}x{img.height}")

    # Resize if too large (match app.py behavior)
    max_dim = 2048
    if max(img.width, img.height) > max_dim:
        ratio = max_dim / max(img.width, img.height)
        new_w = int(img.width * ratio)
        new_h = int(img.height * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"  Resized to: {new_w}x{new_h}")

    results = []

    # --- Phase 1: Pure AI models ---
    print(f"\n  --- Pure AI Models ---")
    ai_sessions = {}  # cache sessions for hybrid reuse

    for model_name in models:
        print(f"\n  Testing: {model_name} ... ", end="", flush=True)

        try:
            session = new_session(model_name, providers=["CPUExecutionProvider"])
            ai_sessions[model_name] = session

            t0 = time.perf_counter()
            raw_result = remove(img, session=session)
            t1 = time.perf_counter()

            elapsed = t1 - t0
            thresholded = apply_alpha_threshold(raw_result)
            green_count, green_pct = analyze_green_remnants(thresholded)

            passed = elapsed < TIME_LIMIT
            status = "PASS" if passed else "SLOW"
            print(f"{elapsed:.2f}s [{status}]  green_px={green_count} ({green_pct:.2f}%)")

            results.append({
                "model": model_name,
                "time": elapsed,
                "passed": passed,
                "image": thresholded,
                "green_count": green_count,
                "green_pct": green_pct,
                "error": None,
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "model": model_name,
                "time": float("inf"),
                "passed": False,
                "image": None,
                "green_count": -1,
                "green_pct": -1,
                "error": str(e),
            })

    # --- Phase 2: Pure colorkey (no AI) ---
    print(f"\n  --- Pure CV Colorkey ---")
    print(f"\n  Testing: colorkey-only ... ", end="", flush=True)
    try:
        t0 = time.perf_counter()
        ck_result = greenscreen_colorkey(img)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        green_count, green_pct = analyze_green_remnants(ck_result)
        passed = elapsed < TIME_LIMIT
        status = "PASS" if passed else "SLOW"
        print(f"{elapsed:.2f}s [{status}]  green_px={green_count} ({green_pct:.2f}%)")
        results.append({
            "model": "colorkey-only",
            "time": elapsed, "passed": passed, "image": ck_result,
            "green_count": green_count, "green_pct": green_pct, "error": None,
        })
    except Exception as e:
        print(f"ERROR: {e}")
        results.append({
            "model": "colorkey-only",
            "time": float("inf"), "passed": False, "image": None,
            "green_count": -1, "green_pct": -1, "error": str(e),
        })

    # --- Phase 3: Hybrid approaches (using fast AI models only) ---
    # Only run hybrids with models that passed or were close to the time limit
    fast_models = [m for m in models if m in ai_sessions]
    # Prioritize the faster ones for hybrid testing
    fast_ai_for_hybrid = []
    for r in results:
        if r["error"] is None and r["model"] in ai_sessions and r["time"] < TIME_LIMIT + 1.0:
            fast_ai_for_hybrid.append(r["model"])

    if not fast_ai_for_hybrid:
        # Fallback: use the fastest available model
        valid = [(r["time"], r["model"]) for r in results if r["error"] is None and r["model"] in ai_sessions]
        if valid:
            valid.sort()
            fast_ai_for_hybrid = [valid[0][1]]

    print(f"\n  --- Hybrid Approaches (AI + CV) ---")
    print(f"  Using AI models for hybrids: {', '.join(fast_ai_for_hybrid)}")

    for ai_model in fast_ai_for_hybrid:
        session = ai_sessions[ai_model]

        # Hybrid 1: AI + green despill
        label = f"{ai_model}+despill"
        print(f"\n  Testing: {label} ... ", end="", flush=True)
        try:
            t0 = time.perf_counter()
            hybrid_result = hybrid_ai_then_despill(img, session)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            green_count, green_pct = analyze_green_remnants(hybrid_result)
            passed = elapsed < TIME_LIMIT
            status = "PASS" if passed else "SLOW"
            print(f"{elapsed:.2f}s [{status}]  green_px={green_count} ({green_pct:.2f}%)")
            results.append({
                "model": label, "time": elapsed, "passed": passed,
                "image": hybrid_result, "green_count": green_count,
                "green_pct": green_pct, "error": None,
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "model": label, "time": float("inf"), "passed": False,
                "image": None, "green_count": -1, "green_pct": -1, "error": str(e),
            })

        # Hybrid 2: AI intersected with colorkey
        label = f"{ai_model}+colorkey"
        print(f"\n  Testing: {label} ... ", end="", flush=True)
        try:
            t0 = time.perf_counter()
            hybrid_result = hybrid_ai_plus_colorkey(img, session)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            green_count, green_pct = analyze_green_remnants(hybrid_result)
            passed = elapsed < TIME_LIMIT
            status = "PASS" if passed else "SLOW"
            print(f"{elapsed:.2f}s [{status}]  green_px={green_count} ({green_pct:.2f}%)")
            results.append({
                "model": label, "time": elapsed, "passed": passed,
                "image": hybrid_result, "green_count": green_count,
                "green_pct": green_pct, "error": None,
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "model": label, "time": float("inf"), "passed": False,
                "image": None, "green_count": -1, "green_pct": -1, "error": str(e),
            })

        # Hybrid 3: AI primary + strict green knockout
        label = f"{ai_model}+greenKO"
        print(f"\n  Testing: {label} ... ", end="", flush=True)
        try:
            t0 = time.perf_counter()
            hybrid_result = hybrid_colorkey_then_ai_refine(img, session)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            green_count, green_pct = analyze_green_remnants(hybrid_result)
            passed = elapsed < TIME_LIMIT
            status = "PASS" if passed else "SLOW"
            print(f"{elapsed:.2f}s [{status}]  green_px={green_count} ({green_pct:.2f}%)")
            results.append({
                "model": label, "time": elapsed, "passed": passed,
                "image": hybrid_result, "green_count": green_count,
                "green_pct": green_pct, "error": None,
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "model": label, "time": float("inf"), "passed": False,
                "image": None, "green_count": -1, "green_pct": -1, "error": str(e),
            })

        # Hybrid 4: AI + GrabCut refinement
        label = f"{ai_model}+grabcut"
        print(f"\n  Testing: {label} ... ", end="", flush=True)
        try:
            t0 = time.perf_counter()
            hybrid_result = hybrid_ai_grabcut(img, session)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            green_count, green_pct = analyze_green_remnants(hybrid_result)
            passed = elapsed < TIME_LIMIT
            status = "PASS" if passed else "SLOW"
            print(f"{elapsed:.2f}s [{status}]  green_px={green_count} ({green_pct:.2f}%)")
            results.append({
                "model": label, "time": elapsed, "passed": passed,
                "image": hybrid_result, "green_count": green_count,
                "green_pct": green_pct, "error": None,
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "model": label, "time": float("inf"), "passed": False,
                "image": None, "green_count": -1, "green_pct": -1, "error": str(e),
            })

        # Hybrid 5: AI + contour cleanup (remove fragments)
        # Uses the pure AI result for this model (already computed)
        ai_result_for_model = None
        for r in results:
            if r["model"] == ai_model and r["image"] is not None:
                ai_result_for_model = r["image"]
                break

        if ai_result_for_model is not None:
            label = f"{ai_model}+contour"
            print(f"\n  Testing: {label} ... ", end="", flush=True)
            try:
                t0 = time.perf_counter()
                # Re-run AI (to get accurate timing) then contour cleanup
                raw = remove(img, session=session)
                cleaned = apply_alpha_threshold(raw)
                cleaned = hybrid_contour_cleanup(cleaned)
                t1 = time.perf_counter()
                elapsed = t1 - t0
                green_count, green_pct = analyze_green_remnants(cleaned)
                passed = elapsed < TIME_LIMIT
                status = "PASS" if passed else "SLOW"
                print(f"{elapsed:.2f}s [{status}]  green_px={green_count} ({green_pct:.2f}%)")
                results.append({
                    "model": label, "time": elapsed, "passed": passed,
                    "image": cleaned, "green_count": green_count,
                    "green_pct": green_pct, "error": None,
                })
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "model": label, "time": float("inf"), "passed": False,
                    "image": None, "green_count": -1, "green_pct": -1, "error": str(e),
                })

    # --- Phase 4: K-Means clustering (pure CV, no AI) ---
    print(f"\n  --- K-Means Color Clustering ---")
    print(f"\n  Testing: kmeans-green ... ", end="", flush=True)
    try:
        t0 = time.perf_counter()
        km_result = hybrid_kmeans_colorkey(img, n_clusters=5)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        green_count, green_pct = analyze_green_remnants(km_result)
        passed = elapsed < TIME_LIMIT
        status = "PASS" if passed else "SLOW"
        print(f"{elapsed:.2f}s [{status}]  green_px={green_count} ({green_pct:.2f}%)")
        results.append({
            "model": "kmeans-green", "time": elapsed, "passed": passed,
            "image": km_result, "green_count": green_count,
            "green_pct": green_pct, "error": None,
        })
    except Exception as e:
        print(f"ERROR: {e}")
        results.append({
            "model": "kmeans-green", "time": float("inf"), "passed": False,
            "image": None, "green_count": -1, "green_pct": -1, "error": str(e),
        })

    return img, results


def save_results(image_path, original, results):
    """Save comparison images and individual results."""
    RESULTS_DIR.mkdir(exist_ok=True)

    stem = Path(image_path).stem
    valid_results = [r for r in results if r["image"] is not None]

    if not valid_results:
        print("  No valid results to save.")
        return

    # Save individual model outputs
    for r in valid_results:
        out_path = RESULTS_DIR / f"{stem}_{r['model']}.png"
        r["image"].save(out_path)

    # Save on-white composites
    for r in valid_results:
        on_white = composite_on_white(r["image"])
        out_path = RESULTS_DIR / f"{stem}_{r['model']}_on_white.png"
        on_white.save(out_path)

    # Save comparison strip
    strip = build_comparison_strip(original, valid_results)
    strip.save(RESULTS_DIR / f"{stem}_comparison.png")

    # Save detail crop
    detail = build_detail_crop(original, valid_results)
    detail.save(RESULTS_DIR / f"{stem}_detail.png")

    print(f"\n  Results saved to {RESULTS_DIR}/")


def print_summary(results):
    """Print a formatted results table."""
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'Time':>7} {'Status':>8} {'Green px':>10} {'Green %':>9}")
    print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*10} {'-'*9}")

    for r in results:
        if r["error"]:
            print(f"  {r['model']:<25} {'ERROR':>7} {'FAIL':>8} {'N/A':>10} {'N/A':>9}")
        else:
            status = "PASS" if r["passed"] else "SLOW"
            print(f"  {r['model']:<25} {r['time']:>6.2f}s {status:>8} {r['green_count']:>10} {r['green_pct']:>8.2f}%")

    # Recommendation
    passing = [r for r in results if r["passed"] and r["error"] is None]
    if passing:
        best = min(passing, key=lambda r: (r["green_pct"], r["time"]))
        print(f"\n  RECOMMENDATION: {best['model']}")
        print(f"    Time: {best['time']:.2f}s | Green remnants: {best['green_count']}px ({best['green_pct']:.2f}%)")
    else:
        print(f"\n  No model passed the {TIME_LIMIT}s time limit.")
        fastest = min(results, key=lambda r: r["time"] if r["error"] is None else float("inf"))
        if fastest["error"] is None:
            print(f"  Fastest was {fastest['model']} at {fastest['time']:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark rembg background removal models")
    parser.add_argument("--image", type=str, help="Path to a specific image to test")
    parser.add_argument("--all-images", action="store_true",
                        help="Test all original images in uploads/ (uses latest 3)")
    parser.add_argument("--models", type=str, nargs="+", help="Specific models to test")
    args = parser.parse_args()

    models = args.models if args.models else MODELS

    print("=" * 70)
    print("  Background Removal Model Benchmark")
    print(f"  Models: {', '.join(models)}")
    print(f"  Time limit: {TIME_LIMIT}s per image")
    print("=" * 70)

    if args.image:
        images = [Path(args.image)]
    elif args.all_images:
        originals = sorted(UPLOADS_DIR.glob("*_original.png"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        images = originals[:3]
        print(f"  Testing {len(images)} most recent images")
    else:
        images = [find_latest_image()]
        print(f"  Using most recent image: {images[0].name}")

    all_results = []
    for img_path in images:
        original, results = benchmark_image(img_path, models)
        save_results(img_path, original, results)
        all_results.extend(results)
        print_summary(results)

    if len(images) > 1:
        print(f"\n{'='*70}")
        print(f"  AGGREGATE SUMMARY ({len(images)} images)")
        print(f"{'='*70}")

        # Average times per model
        from collections import defaultdict
        model_times = defaultdict(list)
        model_green = defaultdict(list)
        for r in all_results:
            if r["error"] is None:
                model_times[r["model"]].append(r["time"])
                model_green[r["model"]].append(r["green_pct"])

        print(f"  {'Model':<25} {'Avg Time':>9} {'Avg Green%':>11} {'Status':>8}")
        print(f"  {'-'*25} {'-'*9} {'-'*11} {'-'*8}")
        for model in models:
            if model in model_times:
                avg_t = np.mean(model_times[model])
                avg_g = np.mean(model_green[model])
                status = "PASS" if avg_t < TIME_LIMIT else "SLOW"
                print(f"  {model:<25} {avg_t:>8.2f}s {avg_g:>10.2f}% {status:>8}")


if __name__ == "__main__":
    main()
