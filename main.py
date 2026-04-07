import os
import re
import cv2
from aos_cv import AOS_CV
from dem    import DEM
import numpy as np
_NUMBERS = re.compile(r"(\d+)")

def _numerical_sort_key(value: str):
    parts = _NUMBERS.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
    
def load_images(file_path):
    images = []
    for filename in sorted(os.listdir(file_path), key=_numerical_sort_key):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(file_path, filename))
            if img is not None:
                images.append(img)
    return images




if __name__ == "__main__":
    WIDTH, HEIGHT = 512, 512
    FOV_DEG       = 50.0
    X_POSITIONS   = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5]
    Y_POSITIONS   = [0] * 11
    ALTITUDE      = 35.0
    CENTER_INDEX  = 5          # virtual camera index
    X_VIRTUAL     = X_POSITIONS[CENTER_INDEX]   # = 0
    Y_VIRTUAL     = Y_POSITIONS[CENTER_INDEX]   # = 0
    images=load_images("input_Image")
    dem_flat = DEM()
    dem_flat.load_obj("zero_plane.obj")        # explicit flat DEM

    renderer = AOS_CV(width=WIDTH, height=HEIGHT, fov_deg=FOV_DEG)
    renderer.load_dem(dem_flat)
    # set_dem_transform mirrors pyaos's aos.setDEMTransform([0, 0, focal_plane])
    renderer.set_dem_transform([0, 0, 0])

    for i, img in enumerate(images):
        renderer.add_view(img, X_POSITIONS[i], Y_POSITIONS[i])
    import time
    start=time.time()
    integral_flat_dem = renderer.render(X_VIRTUAL, Y_VIRTUAL, ALTITUDE)
    end=time.time()
    print(f"Rendering time: {end - start:.2f} seconds")
    print(f"  Integral: shape={integral_flat_dem.shape}  "
          f"range=[{integral_flat_dem.min():.1f}, {integral_flat_dem.max():.1f}]")
    cv2.imshow("Integral Flat DEM", np.clip(integral_flat_dem, 0, 255).astype(np.uint8))
    cv2.waitKey(0)