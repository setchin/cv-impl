import re

with open("aos_cv.py", "r") as f:
    content = f.read()

# Modifying add_view
add_view_old = """    def add_view(self, image: np.ndarray, x: float, y: float) -> None:
        \"\"\"
        Register one input image with its nadir camera ground position.

        Parameters
        ----------
        image : np.ndarray   BGR (or grayscale) image — any dtype.
        x, y  : float        Camera ground position in world units.
                             The altitude is specified at render time.
        \"\"\"
        self._views.append((image.astype(np.float32), float(x), float(y)))"""

add_view_new = """    def add_view(self, image: np.ndarray, x: float = None, y: float = None, pose: np.ndarray = None) -> None:
        \"\"\"
        Register one input image with its camera pose.
        \"\"\"
        self._views.append((image.astype(np.float32), x, y, pose))"""

content = content.replace(add_view_old, add_view_new)

with open("aos_cv.py", "w") as f:
    f.write(content)
