import os
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from aos_cv import AOS_CV
from dem import DEM

_NUMBERS = re.compile(r"(\d+)")

def _numerical_sort_key(value: str):
    parts = _NUMBERS.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def load_images(file_path: str, max_images: int = 11) -> list[np.ndarray]:
    images = []
    for filename in sorted(os.listdir(file_path), key=_numerical_sort_key):
        if filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff")):
            img = cv2.imread(os.path.join(file_path, filename))
            if img is not None:
                # Convert Gazebo Camera (U South, V West) to AOS Camera (U East, V South)
                # by rotating 90 degrees clockwise
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                images.append(img)
                if len(images) == max_images:
                    break
    return images

def to_uint8_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.copy()
    return np.clip(image, 0, 255).astype(np.uint8)

def detect_top_features(integral: np.ndarray, max_corners: int = 10) -> np.ndarray:
    gray = cv2.cvtColor(to_uint8_image(integral), cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=12,
        blockSize=7,
        useHarrisDetector=False,
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float32)
    return corners[:, 0, :].astype(np.float32)

def make_distinct_colors(n: int) -> list[tuple[int, int, int]]:
    if n <= 0:
        return []
    colors = []
    for i in range(n):
        hue = int(180.0 * i / max(n, 1))
        hsv = np.uint8([[[hue, 220, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors

def annotate_integral(
    integral: np.ndarray,
    features: np.ndarray,
    colors: list[tuple[int, int, int]],
) -> np.ndarray:
    canvas = to_uint8_image(integral)
    for i, (u, v) in enumerate(features):
        color = colors[i]
        center = (int(round(u)), int(round(v)))
        cv2.circle(canvas, center, 6, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, str(i), (center[0] + 8, center[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return canvas

def annotate_sources(
    renderer: AOS_CV,
    images: list[np.ndarray],
    x_virtual: float,
    y_virtual: float,
    altitude: float,
    features: np.ndarray,
    colors: list[tuple[int, int, int]],
) -> list[np.ndarray]:
    annotated = [to_uint8_image(img) for img in images]

    for feature_idx, (u, v) in enumerate(features):
        color = colors[feature_idx]
        mapping = renderer.backproject_pixel_to_views(
            x_virtual=x_virtual,
            y_virtual=y_virtual,
            altitude=altitude,
            u=float(u),
            v=float(v),
        )

        for item in mapping:
            if not item["valid"]:
                continue
            view_idx = int(item["view_index"])
            if view_idx >= len(annotated):
                continue
            src_u = float(item["source_u"])
            src_v = float(item["source_v"])
            center = (int(round(src_u)), int(round(src_v)))
            cv2.circle(annotated[view_idx], center, 5, color, 2, cv2.LINE_AA)
            cv2.putText(
                annotated[view_idx],
                str(feature_idx),
                (center[0] + 6, center[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    return annotated

def build_contact_sheet(images: list[np.ndarray], cols: int = 4) -> np.ndarray:
    if not images:
        raise ValueError("No images to compose.")

    h, w = images[0].shape[:2]
    rows = int(np.ceil(len(images) / cols))
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        y0, y1 = r * h, (r + 1) * h
        x0, x1 = c * w, (c + 1) * w
        sheet[y0:y1, x0:x1] = img

    return sheet

def load_poses(xml_path: str, max_poses: int = 11):
    x_pos, y_pos, z_pos = [], [], []
    with open(xml_path, 'r') as f:
        for line in f:
            if "<pose>" in line:
                content = line.strip().replace("<pose>", "").replace("</pose>", "")
                parts = content.split()
                if len(parts) >= 3:
                    gz_x = float(parts[0])
                    gz_y = float(parts[1])
                    gz_z = float(parts[2])
                    # Gazebo World: X North, Y West
                    # AOS World: X East, Y South
                    x_pos.append(-gz_y)
                    y_pos.append(-gz_x)
                    z_pos.append(gz_z)
                if len(x_pos) == max_poses:
                    break
    return x_pos, y_pos, z_pos

def main() -> None:
    width, height = 512, 512
    fov_deg = 50.0
    
    # Load up to 11 images from the thermal folder
    dataset_dir = Path("photo_shoot_tuned")
    images = load_images(str(dataset_dir / "thermal"), max_images=11)
    if not images:
        raise RuntimeError("No input images found in thermal dataset.")
        
    x_positions, y_positions, z_positions = load_poses(str(dataset_dir / "poses.xml"), max_poses=11)
    
    if len(images) != len(x_positions):
        raise RuntimeError(
            f"Expected {len(x_positions)} images, got {len(images)}. "
        )

    center_index = len(images) // 2
    x_virtual = x_positions[center_index]
    y_virtual = y_positions[center_index]
    altitude = z_positions[center_index]

    dem_obj = DEM()
    dem_obj.load_obj("converted_dem.obj")

    renderer = AOS_CV(width=width, height=height, fov_deg=fov_deg)
    renderer.load_dem(dem_obj)
    # The terrain from "random_dem.obj" might already be correctly positioned
    renderer.set_dem_transform([0, 0, 0])

    for i, img in enumerate(images):
        renderer.add_view(img, x_positions[i], y_positions[i])

    integral = renderer.render(x_virtual, y_virtual, altitude)

    features = detect_top_features(integral, max_corners=10)
    colors = make_distinct_colors(len(features))

    integral_annotated = annotate_integral(integral, features, colors)
    source_annotated = annotate_sources(
        renderer=renderer,
        images=images,
        x_virtual=x_virtual,
        y_virtual=y_virtual,
        altitude=altitude,
        features=features,
        colors=colors,
    )

    output_dir = Path("results") / "thermal_backprojection"
    output_dir.mkdir(parents=True, exist_ok=True)

    integral_path = output_dir / "integral_annotated.png"
    cv2.imwrite(str(integral_path), integral_annotated)

    for idx, img in enumerate(source_annotated):
        cv2.imwrite(str(output_dir / f"source_{idx:02d}_annotated.png"), img)

    sheet = build_contact_sheet(source_annotated, cols=4)
    sheet_path = output_dir / "source_contact_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)

    print(f"Detected feature count: {len(features)}")
    for i, (u, v) in enumerate(features):
        print(f"Feature {i}: synthetic pixel = ({u:.2f}, {v:.2f})")

    print(f"Saved: {integral_path}")
    print(f"Saved: {sheet_path}")
    print(f"Per-view images saved to: {output_dir}")

if __name__ == "__main__":
    main()
