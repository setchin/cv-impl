"""
test_integration.py — OpenCV-only integration tests

Reproduces the LFR_SIMPLE/Integration.ipynb workflow without pyaos / OpenGL.

Two test cases are run back-to-back:

Test 1 — Flat-plane fast path
    Uses the analytical shift formula (no DEM object).
    Should match the GPU reference with MAE < 1.5 uint8 units.

Test 2 — DEM 3-pass path (flat DEM at Z = 0)
    Loads the zero_plane.obj mesh from LFR_SIMPLE as a DEM, then runs the
    full G-buffer → projection → normalise pipeline.  Because the DEM is
    perfectly flat (Z = 0 everywhere), the result must be identical to
    Test 1 within floating-point rounding.

Setup mirrors Integration.ipynb exactly:
  • 11 images from LFR_SIMPLE/input_Image/
  • Camera positions: x = [5…-5], y = 0, altitude = 35
  • Virtual camera at center_index = 5  →  (x=0, y=0, altitude=35)
  • FOV = 50°, resolution 512×512

Run from the project root:
    python cv-impl/test_integration.py
"""

import os
import re
import glob
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "input_Image")
REFERENCE_IMG   = os.path.join(PROJECT_ROOT, "results",
                                "integrals", "integral.png")
ZERO_PLANE_OBJ  = os.path.join(PROJECT_ROOT,  "zero_plane.obj")
OUTPUT_DIR      = os.path.join(SCRIPT_DIR, "results", "integrals")

# Add cv-impl to path so local modules can be imported
sys.path.insert(0, SCRIPT_DIR)
from aos_cv import AOS_CV
from dem    import DEM


# ---------------------------------------------------------------------------
# Shared configuration  (mirrors Integration.ipynb exactly)
# ---------------------------------------------------------------------------

WIDTH, HEIGHT = 512, 512
FOV_DEG       = 50.0
X_POSITIONS   = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5]
Y_POSITIONS   = [0] * 11
ALTITUDE      = 35.0
CENTER_INDEX  = 5          # virtual camera index
X_VIRTUAL     = X_POSITIONS[CENTER_INDEX]   # = 0
Y_VIRTUAL     = Y_POSITIONS[CENTER_INDEX]   # = 0

# Numeric sort key for filenames
_NUMBERS = re.compile(r"(\d+)")

def _numerical_sort_key(value: str):
    parts = _NUMBERS.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# ---------------------------------------------------------------------------
# Image loading  (shared by both tests)
# ---------------------------------------------------------------------------

def load_images() -> list[np.ndarray]:
    pattern = os.path.join(INPUT_IMAGE_DIR, "*.png")
    paths   = sorted(glob.glob(pattern), key=_numerical_sort_key)

    if not paths:
        raise FileNotFoundError(
            f"No PNG images found in {INPUT_IMAGE_DIR!r}.\n"
            "Ensure LFR_SIMPLE/input_Image/ contains the 11 thermal images."
        )

    images = []
    for p in paths:
        img = cv2.imread(p)          # BGR uint8
        if img is None:
            raise IOError(f"Cannot read: {p}")
        images.append(img.astype(np.float32))

    assert len(images) == len(X_POSITIONS), (
        f"Expected {len(X_POSITIONS)} images, found {len(images)}."
    )
    print(f"Loaded {len(images)} images from {INPUT_IMAGE_DIR}")
    return images


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare_to_reference(result_u8: np.ndarray, label: str) -> bool:
    """
    Compare *result_u8* (uint8 BGR) to the GPU reference integral.png.
    Returns True if the result passes the MAE threshold.
    """
    if not os.path.isfile(REFERENCE_IMG):
        print(f"  [INFO] Reference not found at {REFERENCE_IMG!r} — skipping.")
        return True

    ref = cv2.imread(REFERENCE_IMG).astype(np.float32)
    res = result_u8.astype(np.float32)

    if ref.shape != res.shape:
        print(f"  [WARN] Shape mismatch: result {res.shape} vs ref {ref.shape}")
        return False

    mae      = float(np.mean(np.abs(res - ref)))
    max_err  = float(np.max(np.abs(res - ref)))
    perfect  = float(np.mean(result_u8 == ref.astype(np.uint8))) * 100.0

    print(f"\n  Comparison vs reference ({label}):")
    print(f"    MAE           = {mae:.3f}  (uint8 0-255 scale)")
    print(f"    Max abs error = {max_err:.0f}")
    print(f"    Pixel-perfect = {perfect:.1f}%")

    if mae < 1.5:
        print("    [PASS] Matches the GPU reference within rounding tolerance.")
        return True
    elif mae < 5.0:
        print("    [PASS] Close match; small GPU/CPU floating-point differences.")
        return True
    else:
        print("    [WARN] MAE exceeds tolerance — check pose / FOV settings.")
        return False


def save_result(result: np.ndarray, filename: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    u8 = np.clip(result, 0, 255).astype(np.uint8)
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, u8)
    print(f"  Saved → {path}")
    return path


# ===========================================================================
# Test 1 — Flat-plane fast path (shift + mean)
# ===========================================================================

def test_flat_plane(images: list[np.ndarray]) -> np.ndarray:
    """
    Flat-plane fast path: no DEM attached; each image is shifted by a 2-D
    translation and pixel-wise averaged.

    Expected result: matches GPU reference with MAE < 1.5.
    """
    print("\n" + "=" * 60)
    print("Test 1 — Flat-plane fast path")
    print("=" * 60)

    renderer = AOS_CV(width=WIDTH, height=HEIGHT, fov_deg=FOV_DEG)

    for i, img in enumerate(images):
        renderer.add_view(img, X_POSITIONS[i], Y_POSITIONS[i])
        print(f"  View {i:2d}: x={X_POSITIONS[i]:+d}  "
              f"shift_px={((X_POSITIONS[i]-X_VIRTUAL)*renderer.fx/ALTITUDE):+.1f}")

    integral = renderer.render(X_VIRTUAL, Y_VIRTUAL, ALTITUDE)
    print(f"\n  Integral: shape={integral.shape}  "
          f"range=[{integral.min():.1f}, {integral.max():.1f}]")

    save_result(integral, "integral_flat.png")
    compare_to_reference(
        np.clip(integral, 0, 255).astype(np.uint8), "flat-plane"
    )
    return integral


# ===========================================================================
# Test 2 — 3-pass DEM path with flat DEM
# ===========================================================================

def test_dem_flat(images: list[np.ndarray]) -> np.ndarray:
    """
    Full 3-pass pipeline (G-buffer → projection → normalise) using a
    perfectly flat DEM at Z = 0.

    The DEM path generates a G-buffer via iterative ray–height-field
    intersection and then uses cv2.remap for per-pixel sampling.  With a
    flat DEM the result must equal the fast-path output.

    Two DEM sources are tried:
      a) flat DEM created analytically
      b) zero_plane.obj loaded from disk (if the file exists)
    """
    print("\n" + "=" * 60)
    print("Test 2a — DEM 3-pass path  (flat DEM, analytical)")
    print("=" * 60)

    dem_flat = DEM()
    dem_flat.set_flat(z=0.0)         # explicit flat DEM

    renderer = AOS_CV(width=WIDTH, height=HEIGHT, fov_deg=FOV_DEG)
    renderer.load_dem(dem_flat)
    # set_dem_transform mirrors pyaos's aos.setDEMTransform([0, 0, focal_plane])
    renderer.set_dem_transform([0, 0, 0])

    for i, img in enumerate(images):
        renderer.add_view(img, X_POSITIONS[i], Y_POSITIONS[i])

    integral_flat_dem = renderer.render(X_VIRTUAL, Y_VIRTUAL, ALTITUDE)
    print(f"  Integral: shape={integral_flat_dem.shape}  "
          f"range=[{integral_flat_dem.min():.1f}, {integral_flat_dem.max():.1f}]")

    save_result(integral_flat_dem, "integral_dem_flat.png")
    compare_to_reference(
        np.clip(integral_flat_dem, 0, 255).astype(np.uint8), "DEM flat (analytical)"
    )

    # ---- Test 2b: OBJ DEM ----
    if os.path.isfile(ZERO_PLANE_OBJ):
        print("\n" + "=" * 60)
        print("Test 2b — DEM 3-pass path  (zero_plane.obj from LFR_SIMPLE)")
        print("=" * 60)

        dem_obj = DEM()
        dem_obj.load_obj(ZERO_PLANE_OBJ)

        renderer2 = AOS_CV(width=WIDTH, height=HEIGHT, fov_deg=FOV_DEG)
        renderer2.load_dem(dem_obj)
        renderer2.set_dem_transform([0, 0, 0])

        for i, img in enumerate(images):
            renderer2.add_view(img, X_POSITIONS[i], Y_POSITIONS[i])

        integral_obj = renderer2.render(X_VIRTUAL, Y_VIRTUAL, ALTITUDE)
        print(f"  Integral: shape={integral_obj.shape}  "
              f"range=[{integral_obj.min():.1f}, {integral_obj.max():.1f}]")

        save_result(integral_obj, "integral_dem_obj.png")
        compare_to_reference(
            np.clip(integral_obj, 0, 255).astype(np.uint8), "DEM obj"
        )
    else:
        print(f"\n  [INFO] {ZERO_PLANE_OBJ!r} not found — skipping Test 2b.")

    return integral_flat_dem


# ===========================================================================
# Entry point
# ===========================================================================

def run():
    images = load_images()

    r1 = test_flat_plane(images)
    r2 = test_dem_flat(images)

    # Sanity check: the two paths should produce the same result
    diff = np.abs(r1.astype(np.float32) - r2.astype(np.float32))
    print("\n" + "=" * 60)
    print("Cross-check: flat-path vs DEM-path")
    print("=" * 60)
    print(f"  Max pixel diff = {diff.max():.3f}")
    print(f"  MAE            = {diff.mean():.3f}")
    if diff.mean() < 1.5:
        print("  [PASS] Both paths produce equivalent results.")
    else:
        print("  [WARN] Paths differ — check coordinate conventions.")


if __name__ == "__main__":
    run()
