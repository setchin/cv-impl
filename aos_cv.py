"""
aos_cv.py — Pure OpenCV / NumPy implementation of Airborne Optical Sectioning.

Flat-plane fast path
--------------------
When no DEM is loaded (or the DEM is flat at Z = 0), each input image is
aligned to the virtual camera view by a simple 2-D sub-pixel translation, then
all images are pixel-wise averaged.  This is an O(N * W * H) operation that
runs fast on CPU.

DEM-aware 3-pass path  (mirrors algorithm.md exactly)
------------------------------------------------------
Pass 1 — G-buffer: for every virtual output pixel cast a ray from the virtual
         camera and find where it intersects the DEM surface.  The result is a
         (H, W, 3) world-XYZ map.

Pass 2 — Projection: for each input camera, project every valid G-buffer world
         point into that camera's image plane and accumulate its colour.

Pass 3 — Normalise: divide each pixel's colour sum by the number of cameras
         that contributed to it (= alpha channel in the GPU version).

Coordinate convention
---------------------
• X right, Y forward (into scene), Z up.
• Cameras sit at (x, y, altitude) with altitude > 0 above the DEM surface.
• DEM heights are measured from Z = 0 (the local datum).
• The virtual-to-world formula for nadir cameras is:
      X = x_v + (u - cx) / fx · altitude
      Y = y_v - (v - cy) / fy · altitude      ← note the minus: v↓, Y↑
      Z = DEM(X, Y)
• The world-to-input-image formula for nadir camera i is:
      u_i = (X - xi) / (altitude - Z) · fx + cx
      v_i = -(Y - yi) / (altitude - Z) · fy + cy
"""

import math
import numpy as np
import cv2
from typing import Any

from dem import DEM


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_perspective(fov_deg: float, aspect: float = 1.0,
                       near: float = 0.1, far: float = 10000.0) -> np.ndarray:
    """
    Build a column-vector perspective projection matrix identical to
    glm::perspective.  Output shape: (4, 4) float64.
    """
    f = 1.0 / math.tan(math.radians(fov_deg / 2.0))
    return np.array([
        [f / aspect, 0,                            0,                              0],
        [0,          f,                            0,                              0],
        [0,          0, (far + near) / (near - far), 2 * far * near / (near - far)],
        [0,          0,                           -1,                              0],
    ], dtype=np.float64)


def _project_world_to_pixel(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
    xi: float, yi: float, altitude: float,
    fx: float, fy: float, cx: float, cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast nadir-camera projection (no matrix multiply needed).

    Returns (u_i, v_i) float arrays of the same shape as X/Y/Z; values outside
    [0, W) × [0, H) indicate the world point is not visible in camera i.
    """
    dz = altitude - Z          # depth > 0 for points below the camera
    safe = np.where(dz > 1e-6, dz, 1e-6)
    u_i = (X - xi) / safe * fx + cx
    v_i = -(Y - yi) / safe * fy + cy
    return u_i.astype(np.float32), v_i.astype(np.float32)


def _bilinear_sample(
    img: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample *img* at floating-point pixel coordinates (map_x, map_y).

    Returns (sampled_image, sample_valid_mask) both of shape (H_out, W_out, C)
    and (H_out, W_out) respectively.

    Only pixels where *valid_mask* is True AND the coordinate is inside the
    source image contribute; others are zero.
    """
    H, W = map_x.shape
    # cv2.remap handles OOB with BORDER_CONSTANT = 0
    if img.ndim == 2:
        img3 = img[:, :, np.newaxis]
    else:
        img3 = img
    C = img3.shape[2]

    # Build full remap arrays (zeros for invalid pixels)
    full_x = np.where(valid_mask, map_x, -1.0).astype(np.float32)
    full_y = np.where(valid_mask, map_y, -1.0).astype(np.float32)

    sampled = cv2.remap(
        img3, full_x, full_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if sampled.ndim == 2:
        sampled = sampled[:, :, np.newaxis]

    # A ones-image tells us which output pixels actually came from within the source
    ones = np.ones(img3.shape[:2], dtype=np.float32)
    valid_src = cv2.remap(
        ones, full_x, full_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    sample_ok = valid_mask & (valid_src > 0.5)
    return sampled.astype(np.float32), sample_ok


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AOS_CV:
    """
    CPU-only Airborne Optical Sectioning renderer.

    Nadir-camera API (matches Integration.ipynb)
    -------------------------------------------
    ::

        renderer = AOS_CV(512, 512, fov_deg=50.0)
        renderer.add_view(image, x=5,  y=0)   # camera at (5, 0, altitude)
        renderer.add_view(image, x=0,  y=0)   # ...
        result = renderer.render(x_virtual=0, y_virtual=0, altitude=35)

    DEM-aware API
    -------------
    ::

        dem = DEM()
        dem.load_heightmap("terrain.png", x_min=-50, x_max=50, ...)
        renderer.load_dem(dem)
        renderer.set_dem_transform([0, 0, 0])   # optional translation

        # same add_view / render calls as above
        result = renderer.render(x_virtual=0, y_virtual=0, altitude=35)

    Parameters
    ----------
    width, height : int      Output resolution.
    fov_deg       : float    Horizontal/vertical field of view (square images).
    n_iters       : int      Iterations for the ray–height-field intersection
                             in the G-buffer pass (3–5 is usually sufficient).
    """

    def __init__(
        self,
        width:   int   = 512,
        height:  int   = 512,
        fov_deg: float = 50.0,
        n_iters: int   = 5,
    ):
        self.width   = width
        self.height  = height
        self.fov_deg = fov_deg
        self.n_iters = n_iters

        # Derived intrinsics
        self._f_ndc = 1.0 / math.tan(math.radians(fov_deg / 2.0))
        self._fx    = self._f_ndc * (width  / 2.0)
        self._fy    = self._f_ndc * (height / 2.0)
        self._cx    = width  / 2.0
        self._cy    = height / 2.0

        # View registry: list of (image: float32, xi: float, yi: float)
        self._views: list = []

        # DEM (optional; None → flat Z = 0)
        self._dem: DEM | None = None

    # ------------------------------------------------------------------
    # View management
    # ------------------------------------------------------------------

    def add_view(self, image: np.ndarray, x: float, y: float) -> None:
        """
        Register one input image with its nadir camera ground position.

        Parameters
        ----------
        image : np.ndarray   BGR (or grayscale) image — any dtype.
        x, y  : float        Camera ground position in world units.
                             The altitude is specified at render time.
        """
        self._views.append((image.astype(np.float32), float(x), float(y)))

    def clear_views(self) -> None:
        """Remove all registered views."""
        self._views.clear()

    # ------------------------------------------------------------------
    # DEM management
    # ------------------------------------------------------------------

    def load_dem(self, dem: DEM) -> None:
        """
        Attach a DEM to the renderer.  Subsequent :meth:`render` calls will use
        the full 3-pass pipeline instead of the flat-plane shortcut.

        Pass ``None`` to revert to the flat-plane fast path.
        """
        self._dem = dem

    def set_dem_transform(self, translation=(0.0, 0.0, 0.0)) -> None:
        """
        Translate the DEM in world space (mirrors pyaos ``setDEMTransform``).
        Has no effect when no DEM is loaded.
        """
        if self._dem is not None:
            self._dem.set_transform(translation)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(
        self,
        x_virtual:  float,
        y_virtual:  float,
        altitude:   float,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """
        Compute the AOS integral image.

        Selects the fast flat-plane path when no DEM is loaded or the DEM is
        flat, and the full 3-pass DEM path otherwise.

        Parameters
        ----------
        x_virtual, y_virtual : float   Virtual camera ground position.
        altitude             : float   Camera height above Z = 0.
        interpolation        : int     OpenCV interpolation flag.

        Returns
        -------
        integral : np.ndarray, float32, shape (H, W, C)
            Pixel-wise average across all registered views.
        """
        if not self._views:
            raise ValueError("No views registered. Call add_view() first.")

        # Choose rendering path
        use_dem = (
            self._dem is not None
            and self._dem._type != "flat"
        )

        if use_dem:
            return self._render_dem(
                x_virtual, y_virtual, altitude, interpolation
            )
        else:
            return self._render_flat(
                x_virtual, y_virtual, altitude, interpolation
            )

    def backproject_pixel_to_views(
        self,
        x_virtual: float,
        y_virtual: float,
        altitude: float,
        u: float,
        v: float,
    ) -> list[dict[str, Any]]:
        """
        Backproject one synthetic-image pixel to every registered source view.

        Parameters
        ----------
        x_virtual, y_virtual : float
            Virtual camera ground position used when rendering.
        altitude : float
            Camera altitude used when rendering.
        u, v : float
            Pixel coordinate in the synthetic image (x, y / column, row).

        Returns
        -------
        list[dict]
            One entry per source image with fields:
            - view_index: int
            - source_u: float
            - source_v: float
            - valid: bool (inside source image bounds)
            - world_xyz: tuple[float, float, float]
        """
        if not self._views:
            raise ValueError("No views registered. Call add_view() first.")

        Xw, Yw, Zw = self._virtual_pixel_to_world(
            x_virtual=x_virtual,
            y_virtual=y_virtual,
            altitude=altitude,
            u=u,
            v=v,
        )

        results: list[dict[str, Any]] = []
        for idx, (img, xi, yi) in enumerate(self._views):
            u_i, v_i = _project_world_to_pixel(
                np.array([[Xw]], dtype=np.float32),
                np.array([[Yw]], dtype=np.float32),
                np.array([[Zw]], dtype=np.float32),
                xi,
                yi,
                altitude,
                self._fx,
                self._fy,
                self._cx,
                self._cy,
            )
            src_u = float(u_i[0, 0])
            src_v = float(v_i[0, 0])
            valid = (0.0 <= src_u < img.shape[1]) and (0.0 <= src_v < img.shape[0])

            results.append({
                "view_index": idx,
                "source_u": src_u,
                "source_v": src_v,
                "valid": bool(valid),
                "world_xyz": (float(Xw), float(Yw), float(Zw)),
            })

        return results

    def _virtual_pixel_to_world(
        self,
        x_virtual: float,
        y_virtual: float,
        altitude: float,
        u: float,
        v: float,
    ) -> tuple[float, float, float]:
        """
        Convert one virtual image pixel (u, v) to a world-space point.

        For flat DEM this is analytical. For non-flat DEM, it uses the same
        fixed-point iteration as the G-buffer path.
        """
        dx = (float(u) - self._cx) / self._fx
        dy = -((float(v) - self._cy) / self._fy)

        use_dem = (self._dem is not None and self._dem._type != "flat")

        if use_dem:
            t = float(altitude)
            for _ in range(self.n_iters):
                x_cur = float(x_virtual + dx * t)
                y_cur = float(y_virtual + dy * t)
                z_dem = float(self._dem.get_z(x_cur, y_cur))
                t = float(altitude - z_dem)
            z_world = float(altitude - t)
        else:
            t = float(altitude)
            z_world = float(self._dem._flat_z + self._dem._translation[2]) if self._dem is not None else 0.0

        x_world = float(x_virtual + dx * t)
        y_world = float(y_virtual + dy * t)
        return x_world, y_world, z_world

    # ------------------------------------------------------------------
    # Fast flat-plane path  (shift + mean)
    # ------------------------------------------------------------------

    def _render_flat(
        self,
        x_virtual: float,
        y_virtual: float,
        altitude:  float,
        interpolation: int,
    ) -> np.ndarray:
        """
        Flat-plane optimisation: align each image by a 2-D affine translation
        and average pixel-wise.

        Shift formula (from ``algorithm.md`` CPU derivation):
            tx[i] = (xi - x_virtual) * fx / altitude
            ty[i] = -(yi - y_virtual) * fy / altitude
        """
        sample = self._views[0][0]
        C = 1 if sample.ndim == 2 else sample.shape[2]

        accumulator = np.zeros((self.height, self.width, C), dtype=np.float64)
        weight_map  = np.zeros((self.height, self.width),    dtype=np.float64)

        M    = np.eye(2, 3, dtype=np.float32)
        ones = np.ones((self.height, self.width), dtype=np.float32)

        for img, xi, yi in self._views:
            tx = (xi - x_virtual) * self._fx / altitude
            ty = -(yi - y_virtual) * self._fy / altitude

            M[0, 2] = tx
            M[1, 2] = ty

            if img.ndim == 2:
                img3d = img[:, :, np.newaxis]
            else:
                img3d = img

            shifted = cv2.warpAffine(
                img3d, M, (self.width, self.height),
                flags=interpolation,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            if shifted.ndim == 2:
                shifted = shifted[:, :, np.newaxis]

            valid = cv2.warpAffine(
                ones, M, (self.width, self.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            valid_bin = (valid > 0.5).astype(np.float64)

            accumulator += shifted * valid_bin[:, :, np.newaxis]
            weight_map  += valid_bin

        safe_w   = np.maximum(weight_map, 1.0)[:, :, np.newaxis]
        integral = (accumulator / safe_w).astype(np.float32)
        integral[weight_map == 0] = 0.0
        return integral

    # ------------------------------------------------------------------
    # 3-pass DEM path  (mirrors algorithm.md GPU pipeline)
    # ------------------------------------------------------------------

    def _render_dem(
        self,
        x_virtual: float,
        y_virtual: float,
        altitude:  float,
        interpolation: int,
    ) -> np.ndarray:
        """
        Full 3-pass pipeline (algorithm.md):

        Pass 1 — G-buffer  : iterative ray–DEM intersection.
        Pass 2 — Projection: for each view, project G-buffer world points into
                             the input image and accumulate colour.
        Pass 3 — Normalise : divide by overlap count.
        """
        # ---- Pass 1 — G-buffer ----------------------------------------
        gbuffer, valid = self._build_gbuffer(x_virtual, y_virtual, altitude)
        # gbuffer: (H, W, 3) float32, world XYZ at each pixel
        # valid:   (H, W) bool

        # ---- Pass 2 — Projection + accumulation -----------------------
        sample  = self._views[0][0]
        C       = 1 if sample.ndim == 2 else sample.shape[2]

        accumulator = np.zeros((self.height, self.width, C), dtype=np.float64)
        weight_map  = np.zeros((self.height, self.width),    dtype=np.float64)

        X_gb = gbuffer[:, :, 0]   # (H, W)
        Y_gb = gbuffer[:, :, 1]
        Z_gb = gbuffer[:, :, 2]

        for img, xi, yi in self._views:
            # Project world positions → input camera i pixel coordinates
            u_i, v_i = _project_world_to_pixel(
                X_gb, Y_gb, Z_gb,
                xi, yi, altitude,
                self._fx, self._fy, self._cx, self._cy,
            )

            sampled, sample_ok = _bilinear_sample(img, u_i, v_i, valid)

            accumulator += sampled.astype(np.float64) * sample_ok[:, :, np.newaxis]
            weight_map  += sample_ok.astype(np.float64)

        # ---- Pass 3 — Normalise ---------------------------------------
        safe_w   = np.maximum(weight_map, 1.0)[:, :, np.newaxis]
        integral = (accumulator / safe_w).astype(np.float32)
        integral[weight_map == 0] = 0.0
        return integral

    # ------------------------------------------------------------------
    # G-buffer builder  (Pass 1 implementation)
    # ------------------------------------------------------------------

    def _build_gbuffer(
        self,
        x_virtual: float,
        y_virtual: float,
        altitude:  float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Iterative ray–height-field intersection for every output pixel.

        For each pixel (u, v) the ray from the virtual nadir camera is:
            origin    = (x_virtual, y_virtual, altitude)
            direction = ((u - cx)/fx,  -(v - cy)/fy,  -1)   [un-normalised]

        Parametric ray:
            X(t) = x_v + dx * t
            Y(t) = y_v + dy * t
            Z(t) = altitude - t

        Iteration:
            t_0   = altitude         (flat-plane seed: Z_dem = 0)
            t_k+1 = altitude - DEM(X(t_k), Y(t_k))

        Converges in 2–3 steps for typical terrain slopes.

        Returns
        -------
        gbuffer : (H, W, 3) float32 — world XYZ at each pixel
        valid   : (H, W) bool       — True where the ray hit the DEM
        """
        H, W = self.height, self.width

        # Pixel centres → horizontal ray slopes
        u_grid = np.arange(W, dtype=np.float32)                  # (W,)
        v_grid = np.arange(H, dtype=np.float32)                  # (H,)
        U, V   = np.meshgrid(u_grid, v_grid)                     # (H, W)

        dx = (U - self._cx) / self._fx    # (H, W)
        dy = -((V - self._cy) / self._fy) # (H, W)  minus: v↓, y↑

        # Initialise with flat-plane seed (t = altitude → Z_dem = 0)
        t = np.full((H, W), altitude, dtype=np.float32)

        for _ in range(self.n_iters):
            X_cur = x_virtual + dx * t     # (H, W)
            Y_cur = y_virtual + dy * t
            Z_dem = self._dem.get_z(X_cur, Y_cur)     # (H, W)
            t     = (altitude - Z_dem).astype(np.float32)

        # Final world positions
        X_world = (x_virtual + dx * t).astype(np.float32)
        Y_world = (y_virtual + dy * t).astype(np.float32)
        Z_world = (altitude  - t     ).astype(np.float32)

        gbuffer          = np.stack([X_world, Y_world, Z_world], axis=2)
        # Mark pixels where the ray hit a valid DEM region
        valid = np.isfinite(Z_world) & (t > 0)

        return gbuffer, valid

    # ------------------------------------------------------------------
    # Intrinsic properties
    # ------------------------------------------------------------------

    @property
    def f_ndc(self) -> float:
        return self._f_ndc

    @property
    def fx(self) -> float:
        return self._fx

    @property
    def fy(self) -> float:
        return self._fy
