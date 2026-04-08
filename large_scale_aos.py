import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import time
from aos_cv import AOS_CV
from dem import DEM

def main():
    rgb_dir = "photo_shoot_tuned/rgb"
    thermal_dir = "photo_shoot_tuned/thermal"
    poses_file = "photo_shoot_tuned/poses.xml"
    dem_file = "photo_shoot_tuned/random_dem.obj"
    out_dir = "results/simulation"

    # Make output directory
    os.makedirs(out_dir, exist_ok=True)

    # 1. Parse poses
    print("Parsing poses...")
    tree = ET.parse(poses_file)
    root = tree.getroot()
    poses = []
    # format: <pose>x y z roll pitch yaw</pose>
    for pose in root.findall('pose'):
        parts = pose.text.strip().split()
        if len(parts) >= 3:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            poses.append((x, y, z))
            
    # Calculate covered bounding box
    xs = [p[0] for p in poses]
    ys = [p[1] for p in poses]
    
    # 2. Setup DEM
    print("Loading DEM...")
    dem = DEM()
    dem.load_obj(dem_file)
    
    # 3. Setup AOS Parameters
    # Target virtual camera at (0, 0, 100)
    VIRTUAL_X = 0.0
    VIRTUAL_Y = 0.0
    VIRTUAL_ALTITUDE = 100.0
    
    # We want a FOV large enough to cover the terrain bounds
    max_d = max(abs(min(xs)), abs(max(xs)), abs(min(ys)), abs(max(ys)))
    # We need: tan(FOV/2) = max_d / VIRTUAL_ALTITUDE
    min_fov_rad = np.arctan(max_d / VIRTUAL_ALTITUDE)
    FOV = float(2.0 * np.degrees(min_fov_rad)) * 1.5  # Add a 50% margin
    
    print(f"Using FOV = {FOV:.1f} degrees to cover terraian spanning {max_d}m from altitude {VIRTUAL_ALTITUDE}m")

    # High resolution
    WIDTH, HEIGHT = 2048, 2048

    def render_modality(modality_dir, modality_name):
        print(f"--- Processing {modality_name} ---")
        renderer = AOS_CV(width=WIDTH, height=HEIGHT, fov_deg=FOV)
        renderer.load_dem(dem)
        # Use default transform at local coordinates
        renderer.set_dem_transform([0, 0, 0])
        
        # Load views
        loaded_count = 0
        for i, (x, y, z) in enumerate(poses):
            img_path = os.path.join(modality_dir, f"pose_{i}_{modality_name}.png")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    renderer.add_view(img, x, y)
                    loaded_count += 1
        
        print(f"Loaded {loaded_count} images for {modality_name} AOS.")
        
        print(f"Starting {modality_name} rendering...")
        t0 = time.time()
        # AOS_CV interface render accepts a single altitude value for projection.
        res = renderer.render(VIRTUAL_X, VIRTUAL_Y, VIRTUAL_ALTITUDE)
        t1 = time.time()
        print(f"Rendered {modality_name} in {t1-t0:.2f} seconds")
        
        out_path = os.path.join(out_dir, f"large_scale_aos_{modality_name}.png")
        cv2.imwrite(out_path, np.clip(res, 0, 255).astype(np.uint8))
        print(f"Saved to {out_path}\n")

    # Render for both RGB and Thermal
    render_modality(rgb_dir, "rgb")
    render_modality(thermal_dir, "thermal")

if __name__ == '__main__':
    main()
