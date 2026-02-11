"""
Iterative Background Removal Optimizer
=======================================
Captures from webcam, tests multiple edge-aware removal strategies,
measures both green remnants AND edge smoothness, and saves comparison
results for each iteration.

The problem: hard colorkey intersection creates jagged edges where
green screen has shadow gradients. We need soft-edge blending.

Usage:
    venv/Scripts/python.exe test_iterate.py
    venv/Scripts/python.exe test_iterate.py --rounds 5
    venv/Scripts/python.exe test_iterate.py --image uploads/some_image.png

Results saved to test_iterations/ folder.
"""

import os
import sys
import time
import argparse
from pathlib import Path

os.environ["ORT_TENSORRT_UNAVAILABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import onnxruntime
onnxruntime.set_default_logger_severity(3)
from rembg import remove, new_session

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("test_iterations")
TIME_LIMIT = 2.0
ALPHA_THRESHOLD = 128

# Pre-load the fast AI session once
print("Loading u2netp model (CPU only)...")
SESSION_U2NETP = new_session("u2netp", providers=["CPUExecutionProvider"])
print("Loading isnet-general-use model (CPU only)...")
SESSION_ISNET = new_session("isnet-general-use", providers=["CPUExecutionProvider"])
print("Models loaded.")


# ---------------------------------------------------------------------------
# Edge quality analysis
# ---------------------------------------------------------------------------

def measure_edge_jaggedness(alpha_channel):
    """
    Measure how jagged/pixelated the edges are.
    Lower = smoother edges. Compares the edge to a Gaussian-smoothed version.

    Returns:
        jaggedness_score: average absolute difference between raw edge
                          and smoothed edge (0 = perfectly smooth)
        edge_pixel_count: number of edge pixels found
    """
    # Find edge pixels using Canny
    edges = cv2.Canny(alpha_channel, 50, 150)
    edge_count = int(np.sum(edges > 0))

    if edge_count == 0:
        return 0.0, 0

    # Compare the alpha along edges to a smoothed version
    smooth_alpha = cv2.GaussianBlur(alpha_channel.astype(np.float32), (5, 5), 1.5)

    # Only look at edge pixels
    edge_mask = edges > 0
    raw_vals = alpha_channel[edge_mask].astype(np.float32)
    smooth_vals = smooth_alpha[edge_mask]

    # Jaggedness = how much the raw edge differs from smooth
    # For a perfectly smooth edge, these would be identical
    jaggedness = float(np.mean(np.abs(raw_vals - smooth_vals)))

    return jaggedness, edge_count


def measure_edge_roughness(alpha_channel):
    """
    Alternative metric: measure the total perimeter vs the convex hull perimeter.
    Rough/jagged edges have much higher perimeter than their convex hull.

    Returns roughness_ratio (1.0 = perfectly smooth convex shape, higher = rougher)
    """
    contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0

    largest = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest, True)
    hull = cv2.convexHull(largest)
    hull_perimeter = cv2.arcLength(hull, True)

    if hull_perimeter == 0:
        return 1.0

    return perimeter / hull_perimeter


def analyze_green_remnants(rgba_img):
    """Count green-dominant pixels in the opaque region."""
    arr = np.array(rgba_img)
    alpha = arr[:, :, 3]
    opaque = alpha > 0

    r = arr[:, :, 0].astype(float)
    g = arr[:, :, 1].astype(float)
    b = arr[:, :, 2].astype(float)

    green_dominant = (g > r + 30) & (g > b + 30) & (g > 80)
    green_opaque = green_dominant & opaque

    total_opaque = int(np.sum(opaque))
    green_count = int(np.sum(green_opaque))
    pct = (green_count / total_opaque * 100) if total_opaque > 0 else 0.0
    return green_count, pct


def full_quality_analysis(rgba_img, label_name):
    """Run all quality metrics on a result image."""
    arr = np.array(rgba_img)
    alpha = arr[:, :, 3]

    green_count, green_pct = analyze_green_remnants(rgba_img)
    jaggedness, edge_px = measure_edge_jaggedness(alpha)
    roughness = measure_edge_roughness(alpha)

    return {
        "name": label_name,
        "green_count": green_count,
        "green_pct": green_pct,
        "jaggedness": jaggedness,
        "edge_pixels": edge_px,
        "roughness": roughness,
    }


# ---------------------------------------------------------------------------
# Removal strategies (ordered from simplest to most sophisticated)
# ---------------------------------------------------------------------------

def strategy_ai_only(img_rgb, session=SESSION_U2NETP):
    """Pure AI with hard alpha threshold. Baseline."""
    raw = remove(img_rgb, session=session)
    arr = np.array(raw)
    alpha = arr[:, :, 3]
    alpha = np.where(alpha < ALPHA_THRESHOLD, 0, 255).astype(np.uint8)
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, "RGBA")


def strategy_ai_soft_alpha(img_rgb, session=SESSION_U2NETP):
    """
    AI removal but preserve the soft alpha edges from the model instead
    of hard thresholding. Only zero out very low alpha (< 30).
    The model's native alpha IS the anti-aliased edge.
    """
    raw = remove(img_rgb, session=session)
    arr = np.array(raw)
    alpha = arr[:, :, 3]
    # Gentle threshold: kill near-zero but preserve gradients
    alpha = np.where(alpha < 30, 0, alpha).astype(np.uint8)
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, "RGBA")


def strategy_soft_colorkey_intersect(img_rgb, session=SESSION_U2NETP):
    """
    AI mask + SOFT colorkey. Instead of binary green detection,
    use a distance-based green score that fades smoothly.
    """
    # AI pass
    raw = remove(img_rgb, session=session)
    ai_arr = np.array(raw)
    ai_alpha = ai_arr[:, :, 3].astype(np.float32) / 255.0

    # Soft green detection: compute "greenness" as continuous 0-1
    img_array = np.array(img_rgb)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Green hue center = 60 in OpenCV (0-180 scale)
    hue_dist = np.minimum(np.abs(h - 60), 180 - np.abs(h - 60))
    # Greenness: 1.0 at perfect green, 0.0 far from green
    hue_score = np.clip(1.0 - hue_dist / 35.0, 0, 1)
    sat_score = np.clip(s / 255.0, 0, 1)
    # Must have some brightness
    val_score = np.clip(v / 100.0, 0, 1)

    greenness = hue_score * sat_score * val_score
    # Invert: 1.0 = definitely NOT green, 0.0 = definitely green
    not_green = 1.0 - greenness

    # Smooth the not-green mask to avoid jagged transitions
    not_green = cv2.GaussianBlur(not_green.astype(np.float32), (5, 5), 1.5)

    # Combine: multiply AI alpha by not-green score
    combined = ai_alpha * not_green
    combined_u8 = (np.clip(combined, 0, 1) * 255).astype(np.uint8)

    # Apply moderate threshold to clean up very faint pixels
    combined_u8 = np.where(combined_u8 < 20, 0, combined_u8).astype(np.uint8)

    result = np.dstack((img_array, combined_u8))
    return Image.fromarray(result, "RGBA")


def strategy_soft_colorkey_despill(img_rgb, session=SESSION_U2NETP):
    """
    Same as soft_colorkey_intersect but also despills green from edges.
    """
    # AI pass
    raw = remove(img_rgb, session=session)
    ai_arr = np.array(raw)
    ai_alpha = ai_arr[:, :, 3].astype(np.float32) / 255.0

    # Soft green detection
    img_array = np.array(img_rgb)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    hue_dist = np.minimum(np.abs(h - 60), 180 - np.abs(h - 60))
    hue_score = np.clip(1.0 - hue_dist / 35.0, 0, 1)
    sat_score = np.clip(s / 255.0, 0, 1)
    val_score = np.clip(v / 100.0, 0, 1)
    greenness = hue_score * sat_score * val_score
    not_green = cv2.GaussianBlur((1.0 - greenness).astype(np.float32), (5, 5), 1.5)

    combined = ai_alpha * not_green
    combined_u8 = (np.clip(combined, 0, 1) * 255).astype(np.uint8)
    combined_u8 = np.where(combined_u8 < 20, 0, combined_u8).astype(np.uint8)

    # Green spill removal on edge zone
    edge_kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(combined_u8, edge_kernel, iterations=2)
    eroded = cv2.erode(combined_u8, edge_kernel, iterations=2)
    edge_mask = (dilated.astype(int) - eroded.astype(int)).clip(0, 255).astype(np.uint8)

    hsv_u8 = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    green_edge = (hsv_u8[:, :, 0] > 30) & (hsv_u8[:, :, 0] < 90) & (edge_mask > 0)
    hsv_u8[green_edge, 1] = (hsv_u8[green_edge, 1] * 0.25).astype(np.uint8)
    cleaned = cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2RGB)

    result = np.dstack((cleaned, combined_u8))
    return Image.fromarray(result, "RGBA")


def strategy_ai_then_green_fade(img_rgb, session=SESSION_U2NETP):
    """
    Use AI soft alpha, then only fade out green pixels proportionally
    to their greenness — doesn't touch non-green edges at all.
    Preserves the AI model's smooth edge work.
    """
    raw = remove(img_rgb, session=session)
    arr = np.array(raw)
    ai_alpha = arr[:, :, 3].astype(np.float32) / 255.0

    # Compute greenness of original pixels
    img_array = np.array(img_rgb)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    hue_dist = np.minimum(np.abs(h - 60), 180 - np.abs(h - 60))
    hue_score = np.clip(1.0 - hue_dist / 30.0, 0, 1)
    sat_score = np.clip((s - 40) / 180.0, 0, 1)
    greenness = hue_score * sat_score

    # For each pixel, reduce alpha by its greenness
    # Non-green pixels: alpha unchanged (preserves smooth AI edges)
    # Green pixels: alpha reduced/zeroed proportionally
    fade_factor = 1.0 - greenness
    adjusted_alpha = ai_alpha * fade_factor

    # Gentle cleanup threshold
    adjusted_alpha = np.where(adjusted_alpha < 0.08, 0, adjusted_alpha)
    alpha_u8 = (np.clip(adjusted_alpha, 0, 1) * 255).astype(np.uint8)

    # Despill green from semi-transparent edge pixels
    hsv_u8 = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    semi_transparent = (alpha_u8 > 0) & (alpha_u8 < 240)
    green_semi = semi_transparent & (hsv_u8[:, :, 0] > 30) & (hsv_u8[:, :, 0] < 90)
    hsv_u8[green_semi, 1] = (hsv_u8[green_semi, 1] * 0.2).astype(np.uint8)
    cleaned = cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2RGB)

    result = np.dstack((cleaned, alpha_u8))
    return Image.fromarray(result, "RGBA")


def strategy_isnet_green_fade(img_rgb):
    """
    ISNet (already 0% green on its own at 1.19s) + green fade for safety.
    This is the "quality reference" — if ISNet already handles it,
    the green fade is just insurance.
    """
    return strategy_ai_then_green_fade(img_rgb, session=SESSION_ISNET)


def strategy_u2netp_aggressive_fade(img_rgb):
    """
    Like green_fade but more aggressive green detection.
    Wider hue range, lower saturation threshold.
    """
    raw = remove(img_rgb, session=SESSION_U2NETP)
    arr = np.array(raw)
    ai_alpha = arr[:, :, 3].astype(np.float32) / 255.0

    img_array = np.array(img_rgb)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Wider hue range for green (catches shadows)
    hue_dist = np.minimum(np.abs(h - 60), 180 - np.abs(h - 60))
    hue_score = np.clip(1.0 - hue_dist / 40.0, 0, 1)  # wider
    sat_score = np.clip((s - 15) / 150.0, 0, 1)  # lower sat threshold
    # Also catch dark greens
    val_score = np.where(v < 40, 0.3, 1.0)
    greenness = hue_score * sat_score * val_score

    fade_factor = 1.0 - np.clip(greenness * 1.5, 0, 1)  # amplified
    adjusted_alpha = ai_alpha * fade_factor
    adjusted_alpha = np.where(adjusted_alpha < 0.08, 0, adjusted_alpha)
    alpha_u8 = (np.clip(adjusted_alpha, 0, 1) * 255).astype(np.uint8)

    # Despill
    hsv_u8 = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    has_alpha = alpha_u8 > 0
    green_px = has_alpha & (hsv_u8[:, :, 0] > 25) & (hsv_u8[:, :, 0] < 95)
    hsv_u8[green_px, 1] = (hsv_u8[green_px, 1] * 0.15).astype(np.uint8)
    cleaned = cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2RGB)

    result = np.dstack((cleaned, alpha_u8))
    return Image.fromarray(result, "RGBA")


# All strategies to test
STRATEGIES = [
    ("u2netp-hard", strategy_ai_only),
    ("u2netp-soft-alpha", strategy_ai_soft_alpha),
    ("u2netp-soft-colorkey", strategy_soft_colorkey_intersect),
    ("u2netp-soft-ck-despill", strategy_soft_colorkey_despill),
    ("u2netp-green-fade", strategy_ai_then_green_fade),
    ("u2netp-aggro-fade", strategy_u2netp_aggressive_fade),
    ("isnet-green-fade", strategy_isnet_green_fade),
]


# ---------------------------------------------------------------------------
# Webcam capture
# ---------------------------------------------------------------------------

def capture_webcam_frame():
    """Capture a single frame from the default webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return None
    # Let camera adjust exposure
    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: Cannot read frame from webcam")
        return None
    # Convert BGR to RGB PIL Image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ---------------------------------------------------------------------------
# Visual output
# ---------------------------------------------------------------------------

def composite_on_white(rgba_img):
    white = Image.new("RGB", rgba_img.size, (255, 255, 255))
    white.paste(rgba_img, mask=rgba_img.split()[3])
    return white


def make_label(text, width, height=36):
    label = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(label)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except (OSError, IOError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = max(4, (width - tw) // 2)
    draw.text((x, 4), text, fill=(255, 255, 255), font=font)
    return label


def build_comparison(original_rgb, results, thumb_w=350):
    """Build a grid comparison image with labels."""
    aspect = original_rgb.height / original_rgb.width
    thumb_h = int(thumb_w * aspect)
    label_h = 36

    n = len(results) + 1  # +1 for original
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    grid_w = cols * thumb_w + (cols - 1) * 4
    grid_h = rows * (thumb_h + label_h) + (rows - 1) * 4

    grid = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))

    # Original
    orig_thumb = original_rgb.resize((thumb_w, thumb_h), Image.LANCZOS)
    lbl = make_label("ORIGINAL", thumb_w, label_h)
    grid.paste(orig_thumb, (0, 0))
    grid.paste(lbl, (0, thumb_h))

    for i, r in enumerate(results):
        col = (i + 1) % cols
        row = (i + 1) // cols
        x = col * (thumb_w + 4)
        y = row * (thumb_h + label_h + 4)

        if r["image"] is not None:
            thumb = composite_on_white(r["image"].resize((thumb_w, thumb_h), Image.LANCZOS))
            grid.paste(thumb, (x, y))

        q = r["quality"]
        txt = (f'{r["name"]} {r["time"]:.2f}s '
               f'G:{q["green_count"]} J:{q["jaggedness"]:.1f} R:{q["roughness"]:.2f}')
        lbl = make_label(txt, thumb_w, label_h)
        grid.paste(lbl, (x, y + thumb_h))

    return grid


# ---------------------------------------------------------------------------
# Main iteration loop
# ---------------------------------------------------------------------------

def run_iteration(img_rgb, iteration_num):
    """Run all strategies on one image and analyze results."""
    print(f"\n{'='*70}")
    print(f"  ITERATION {iteration_num}")
    print(f"  Image size: {img_rgb.width}x{img_rgb.height}")
    print(f"{'='*70}")

    results = []

    for name, strategy_fn in STRATEGIES:
        print(f"  {name:30s} ... ", end="", flush=True)
        try:
            t0 = time.perf_counter()
            result_img = strategy_fn(img_rgb)
            t1 = time.perf_counter()
            elapsed = t1 - t0

            quality = full_quality_analysis(result_img, name)
            passed = elapsed < TIME_LIMIT

            status = "PASS" if passed else "SLOW"
            print(f"{elapsed:.2f}s [{status}]  "
                  f"green={quality['green_count']:>5} ({quality['green_pct']:.2f}%)  "
                  f"jagged={quality['jaggedness']:.1f}  "
                  f"rough={quality['roughness']:.2f}")

            results.append({
                "name": name,
                "time": elapsed,
                "passed": passed,
                "image": result_img,
                "quality": quality,
                "error": None,
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "name": name, "time": float("inf"), "passed": False,
                "image": None, "quality": full_quality_analysis(
                    Image.new("RGBA", (1, 1)), name), "error": str(e),
            })

    return results


def print_results_table(results):
    """Print formatted comparison table."""
    print(f"\n  {'Strategy':<30} {'Time':>6} {'Pass':>5} {'Green':>7} {'Green%':>7} {'Jagged':>7} {'Rough':>6}")
    print(f"  {'-'*30} {'-'*6} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")

    for r in results:
        if r["error"]:
            print(f"  {r['name']:<30} {'ERR':>6} {'N':>5}")
            continue
        q = r["quality"]
        passed = "Y" if r["passed"] else "N"
        print(f"  {r['name']:<30} {r['time']:>5.2f}s {passed:>5} "
              f"{q['green_count']:>7} {q['green_pct']:>6.2f}% "
              f"{q['jaggedness']:>7.1f} {q['roughness']:>6.2f}")

    # Score and rank: lower is better
    # Score = green_pct * 10 + jaggedness * 0.5 + (time if slow penalty)
    valid = [r for r in results if r["error"] is None]
    for r in valid:
        q = r["quality"]
        time_penalty = max(0, r["time"] - TIME_LIMIT) * 50
        r["score"] = q["green_pct"] * 10 + q["jaggedness"] * 0.5 + time_penalty

    valid.sort(key=lambda r: r["score"])
    print(f"\n  RANKING (lower score = better):")
    for i, r in enumerate(valid):
        q = r["quality"]
        marker = " <-- BEST" if i == 0 else ""
        print(f"  #{i+1} {r['name']:<30} score={r['score']:.1f} "
              f"(green={q['green_pct']:.2f}% jagged={q['jaggedness']:.1f} "
              f"time={r['time']:.2f}s){marker}")


def main():
    parser = argparse.ArgumentParser(description="Iterative BG removal optimizer")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Number of webcam capture rounds")
    parser.add_argument("--image", type=str,
                        help="Use a specific image instead of webcam")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("  Iterative Background Removal Optimizer")
    print(f"  Strategies: {len(STRATEGIES)}")
    print(f"  Time limit: {TIME_LIMIT}s")
    print(f"  Metrics: green remnants + edge jaggedness + edge roughness")
    print("=" * 70)

    all_results = []

    for round_num in range(1, args.rounds + 1):
        # Capture or load image
        if args.image:
            img = Image.open(args.image).convert("RGB")
            print(f"\n  Using image: {args.image}")
        else:
            print(f"\n  Capturing from webcam (round {round_num}/{args.rounds})...")
            img = capture_webcam_frame()
            if img is None:
                continue

        # Save the source frame
        src_path = RESULTS_DIR / f"round{round_num:02d}_source.png"
        img.save(src_path)

        # Run all strategies
        results = run_iteration(img, round_num)
        all_results.append((round_num, results))

        # Print table
        print_results_table(results)

        # Save comparison grid
        grid = build_comparison(img, results)
        grid.save(RESULTS_DIR / f"round{round_num:02d}_comparison.png")

        # Save individual results
        for r in results:
            if r["image"] is not None:
                r["image"].save(RESULTS_DIR / f"round{round_num:02d}_{r['name']}.png")
                on_white = composite_on_white(r["image"])
                on_white.save(RESULTS_DIR / f"round{round_num:02d}_{r['name']}_white.png")

        print(f"\n  Round {round_num} saved to {RESULTS_DIR}/")

    # Aggregate summary across all rounds
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  AGGREGATE ACROSS {len(all_results)} ROUNDS")
        print(f"{'='*70}")

        from collections import defaultdict
        scores = defaultdict(list)
        for _, results in all_results:
            for r in results:
                if r["error"] is None and "score" in r:
                    scores[r["name"]].append(r["score"])

        ranked = [(name, np.mean(s)) for name, s in scores.items()]
        ranked.sort(key=lambda x: x[1])
        for i, (name, avg_score) in enumerate(ranked):
            marker = " <-- WINNER" if i == 0 else ""
            print(f"  #{i+1} {name:<30} avg_score={avg_score:.1f}{marker}")


if __name__ == "__main__":
    main()
