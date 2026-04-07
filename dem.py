"""
dem.py — Digital Elevation Model (DEM) for the AOS CPU renderer.

Supported DEM types
-------------------
flat        A flat horizontal plane at a fixed Z value (default Z = 0).
            This is a fast analytical special case.
heightmap   A regular grid of Z values interpolated bilinearly.
            Suitable for typical aerial-survey terrain data.
mesh        An arbitrary triangle mesh loaded from an OBJ file.
            Used for the G-buffer pass via software rasterisation.

Coordinate convention (matches the AOS CPU renderer in aos_cv.py)
------------------------------------------------------------------
• X is right, Y is forward (into the scene), Z is up.
• Cameras sit at (x, y, altitude) where altitude > 0 means above the DEM.
• The default DEM occupies Z ≈ 0 (sea level or local datum).
"""

import os
import math
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# DEM class
# ---------------------------------------------------------------------------

class DEM:
    """
    Holds elevation data and exposes two interfaces:

    1. get_z(X, Y)   — query height at arbitrary world (X, Y) positions.
                       Needed by the iterative ray-height-field G-buffer builder.

    2. vertices / faces  — triangle mesh (generated on demand from height map,
                           or loaded directly from OBJ). Needed by the software
                           rasteriser G-buffer builder.

    The optional *transform* translates (and, if needed, rotates) the DEM in
    world space, matching pyaos's ``setDEMTransform([dx, dy, dz])`` call.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self):
        self._type         = "flat"          # 'flat' | 'heightmap' | 'mesh'
        self._flat_z       = 0.0

        # Height-map fields
        self._hm           = None            # (H_dem, W_dem) float32 array
        self._hm_x_min     = -50.0
        self._hm_x_max     =  50.0
        self._hm_y_min     = -50.0
        self._hm_y_max     =  50.0

        # Mesh fields (built from height map or loaded from OBJ)
        self._vertices_raw = None            # (N, 3) before transform
        self._faces        = None            # (M, 3) int32

        # World-space transform (translation only for now, matching pyaos)
        self._translation  = np.zeros(3, dtype=np.float32)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    def set_flat(self, z: float = 0.0) -> None:
        """Flat plane at constant Z = z (default 0)."""
        self._type   = "flat"
        self._flat_z = float(z)
        self._vertices_raw = None
        self._faces        = None

    def load_heightmap(
        self,
        source,
        x_min: float = -50.0,
        x_max: float =  50.0,
        y_min: float = -50.0,
        y_max: float =  50.0,
        z_scale: float = 1.0,
        z_offset: float = 0.0,
    ) -> None:
        """
        Load a height map from a file path (PNG/TIFF) or a 2-D numpy array.

        Parameters
        ----------
        source      : str or np.ndarray — image file or height values (H×W)
        x_min/max   : world X extent of the grid
        y_min/max   : world Y extent of the grid
        z_scale     : height values are multiplied by this factor
        z_offset    : added to every height after scaling
        """
        if isinstance(source, (str, os.PathLike)):
            img = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise IOError(f"Cannot read height map: {source}")
            hm = img.astype(np.float32)
            if hm.ndim == 3:
                hm = hm[:, :, 0]   # take first channel
        else:
            hm = np.asarray(source, dtype=np.float32)
            if hm.ndim != 2:
                raise ValueError("Height map array must be 2-D (H, W).")

        self._hm      = hm * z_scale + z_offset
        self._hm_x_min = float(x_min)
        self._hm_x_max = float(x_max)
        self._hm_y_min = float(y_min)
        self._hm_y_max = float(y_max)
        self._type    = "heightmap"

        # Build mesh for the rasteriser path
        self._build_mesh_from_heightmap()

    def load_obj(self, path: str) -> None:
        """
        Load a triangle mesh from a Wavefront OBJ file.

        Only ``v`` (vertex) and ``f`` (face) lines are parsed.
        Faces are triangulated (fan triangulation for n-gons).
        """
        verts, faces = [], []
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("v "):
                    _, *xyz = line.split()
                    verts.append([float(x) for x in xyz[:3]])
                elif line.startswith("f "):
                    _, *refs = line.split()
                    # Each ref can be "v", "v/vt", or "v/vt/vn" — take first idx
                    idxs = [int(r.split("/")[0]) - 1 for r in refs]
                    # Fan triangulation
                    for k in range(1, len(idxs) - 1):
                        faces.append([idxs[0], idxs[k], idxs[k + 1]])

        if not verts:
            raise ValueError(f"No vertices found in {path!r}")
        if not faces:
            raise ValueError(f"No faces found in {path!r}")

        self._vertices_raw = np.array(verts, dtype=np.float32)
        self._faces        = np.array(faces, dtype=np.int32)
        self._type         = "mesh"

        # Build a height map from the mesh bbox for the ray-cast path
        self._build_heightmap_from_mesh()

    # ------------------------------------------------------------------
    # Transform (matches pyaos setDEMTransform)
    # ------------------------------------------------------------------

    def set_transform(self, translation=(0.0, 0.0, 0.0)) -> None:
        """Translate the DEM in world space."""
        self._translation = np.array(translation, dtype=np.float32)
        if self._type == "mesh":
            self._build_heightmap_from_mesh()   # refresh height-map cache

    # ------------------------------------------------------------------
    # Public geometry access
    # ------------------------------------------------------------------

    @property
    def vertices(self) -> np.ndarray:
        """
        World-space vertices (N, 3) after applying the current transform.
        For flat plane a minimal 4-vertex quad is generated on the fly.
        """
        raw = self._get_raw_vertices()
        return raw + self._translation[np.newaxis, :]

    @property
    def faces(self) -> np.ndarray:
        """Triangle face indices (M, 3)."""
        if self._faces is None:
            self._ensure_mesh()
        return self._faces

    # ------------------------------------------------------------------
    # Height query  (core of the ray-heightfield G-buffer path)
    # ------------------------------------------------------------------

    def get_z(self, X, Y) -> np.ndarray:
        """
        Return DEM height at world (X, Y) positions.

        Parameters
        ----------
        X, Y : scalar or np.ndarray (arbitrary shape, must match)

        Returns
        -------
        Z : same shape as X/Y, float32
        """
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        tx, ty, tz = self._translation

        if self._type == "flat":
            return np.full_like(X, self._flat_z + tz)

        if self._hm is None:
            raise RuntimeError("Height map not available — call load_heightmap() first.")

        # Shift query by DEM translation
        Xs = X - tx
        Ys = Y - ty

        H_dem, W_dem = self._hm.shape
        # Normalised [0, 1] coordinates over the grid extent
        u = (Xs - self._hm_x_min) / (self._hm_x_max - self._hm_x_min) * (W_dem - 1)
        v = (Ys - self._hm_y_min) / (self._hm_y_max - self._hm_y_min) * (H_dem - 1)

        # Vectorised bilinear interpolation (no cv2.remap size limit)
        u = np.clip(u, 0, W_dem - 1)
        v = np.clip(v, 0, H_dem - 1)

        u0 = np.floor(u).astype(int)
        v0 = np.floor(v).astype(int)
        u1 = np.minimum(u0 + 1, W_dem - 1)
        v1 = np.minimum(v0 + 1, H_dem - 1)

        # Fractional parts
        wu = (u - u0).astype(np.float32)
        wv = (v - v0).astype(np.float32)

        z00 = self._hm[v0, u0]
        z10 = self._hm[v0, u1]
        z01 = self._hm[v1, u0]
        z11 = self._hm[v1, u1]

        z_raw = (z00 * (1 - wu) * (1 - wv)
               + z10 * wu        * (1 - wv)
               + z01 * (1 - wu) * wv
               + z11 * wu        * wv)

        return z_raw + tz

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_raw_vertices(self) -> np.ndarray:
        self._ensure_mesh()
        return self._vertices_raw

    def _ensure_mesh(self) -> None:
        """Generate mesh geometry if not already available."""
        if self._vertices_raw is not None:
            return
        if self._type == "flat":
            self._build_flat_mesh()
        elif self._type == "heightmap":
            self._build_mesh_from_heightmap()
        else:
            raise RuntimeError("Mesh geometry missing — call load_obj() first.")

    def _build_flat_mesh(self, half_size: float = 200.0) -> None:
        """Four-vertex quad centred at origin at Z = flat_z."""
        z = self._flat_z
        s = half_size
        self._vertices_raw = np.array([
            [-s, -s, z],
            [ s, -s, z],
            [ s,  s, z],
            [-s,  s, z],
        ], dtype=np.float32)
        self._faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    def _build_mesh_from_heightmap(self) -> None:
        """Tesselate the height map into a regular triangle grid."""
        H_dem, W_dem = self._hm.shape
        # Down-sample large height maps to avoid millions of triangles
        max_grid = 256
        step_x = max(1, W_dem // max_grid)
        step_y = max(1, H_dem // max_grid)

        xs = np.linspace(self._hm_x_min, self._hm_x_max,
                         (W_dem - 1) // step_x + 1)
        ys = np.linspace(self._hm_y_min, self._hm_y_max,
                         (H_dem - 1) // step_y + 1)
        gx, gy = np.meshgrid(xs, ys)         # (nY, nX)
        gz = cv2.resize(self._hm, (len(xs), len(ys)),
                        interpolation=cv2.INTER_LINEAR)

        nY, nX = gz.shape
        verts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1
                         ).astype(np.float32)

        # Build quad-split triangles
        row_idx = np.arange(nY - 1)
        col_idx = np.arange(nX - 1)
        r, c = np.meshgrid(row_idx, col_idx, indexing="ij")
        r, c = r.ravel(), c.ravel()

        v00 = r * nX + c
        v10 = v00 + 1
        v01 = v00 + nX
        v11 = v01 + 1

        faces = np.stack([
            np.stack([v00, v10, v11], axis=1),
            np.stack([v00, v11, v01], axis=1),
        ], axis=1).reshape(-1, 3).astype(np.int32)

        self._vertices_raw = verts
        self._faces        = faces

    def _build_heightmap_from_mesh(self, grid_size: int = 256) -> None:
        """Rasterise a mesh DEM to a height map for fast get_z queries."""
        verts = self._vertices_raw + self._translation
        xs, ys, zs = verts[:, 0], verts[:, 1], verts[:, 2]

        self._hm_x_min = float(xs.min())
        self._hm_x_max = float(xs.max())
        self._hm_y_min = float(ys.min())
        self._hm_y_max = float(ys.max())

        hm = np.zeros((grid_size, grid_size), dtype=np.float32)
        count = np.zeros_like(hm)

        # Scatter vertex Z values onto the grid
        u = ((xs - self._hm_x_min) / max(self._hm_x_max - self._hm_x_min, 1e-6)
             * (grid_size - 1)).astype(int)
        v = ((ys - self._hm_y_min) / max(self._hm_y_max - self._hm_y_min, 1e-6)
             * (grid_size - 1)).astype(int)
        np.clip(u, 0, grid_size - 1, out=u)
        np.clip(v, 0, grid_size - 1, out=v)

        for i in range(len(verts)):
            hm[v[i], u[i]] += zs[i]
            count[v[i], u[i]] += 1

        mask = count > 0
        hm[mask] /= count[mask]

        # Fill empty cells with nearest-filled values
        dist, idx = cv2.distanceTransformWithLabels(
            (~mask).astype(np.uint8), cv2.DIST_L2, 5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        hm[~mask] = hm.flat[idx[~mask] - 1]
        hm = cv2.GaussianBlur(hm, (3, 3), 0)   # smooth the scatter

        self._hm = hm
        # Reset translation since we already applied it above
        old_t = self._translation.copy()
        self._translation = np.zeros(3, dtype=np.float32)
        self._translation = old_t
