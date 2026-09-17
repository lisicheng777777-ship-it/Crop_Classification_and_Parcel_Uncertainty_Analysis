"""Spatial patch extraction aligned to distance-metadata sample rows."""

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm


def extract_spatial_patches(raster_path, metadata, n_channels, patch_size=3):
    required = {"x", "y"}
    if not required.issubset(metadata.columns):
        raise ValueError("STSMamba metadata requires x and y columns")
    radius = patch_size // 2
    with rasterio.open(raster_path) as src:
        if src.count % n_channels != 0:
            raise ValueError("Raster band count is incompatible with n_channels")
        time_steps = src.count // n_channels
        patches = np.empty(
            (len(metadata), time_steps, patch_size, patch_size, n_channels),
            dtype=np.float32,
        )
        nodata = src.nodata
        for output_index, row in enumerate(
            tqdm(metadata.itertuples(index=False), total=len(metadata), desc="Extracting STSMamba patches")
        ):
            raster_row, raster_col = src.index(float(row.x), float(row.y))
            window = Window(
                raster_col - radius, raster_row - radius, patch_size, patch_size
            )
            raw = src.read(window=window, boundless=True, fill_value=nodata).astype(np.float32)
            raw = raw.reshape(time_steps, n_channels, patch_size, patch_size).transpose(0, 2, 3, 1)
            center = raw[:, radius, radius, :].copy()
            invalid = ~np.isfinite(raw)
            if nodata is not None:
                invalid |= np.isclose(raw, nodata)
            raw = np.where(invalid, center[:, None, None, :], raw)
            patches[output_index] = raw
    return patches


def window_spatial_patches(raw_bands, n_channels, patch_size):
    """Convert a halo-padded [bands,H+2r,W+2r] block to [H*W,T,P,P,C]."""
    radius = patch_size // 2
    bands, padded_h, padded_w = raw_bands.shape
    time_steps = bands // n_channels
    height, width = padded_h - 2 * radius, padded_w - 2 * radius
    cube = raw_bands.reshape(time_steps, n_channels, padded_h, padded_w).transpose(2, 3, 0, 1)
    parts = []
    for dy in range(patch_size):
        row_parts = []
        for dx in range(patch_size):
            row_parts.append(cube[dy:dy + height, dx:dx + width])
        parts.append(np.stack(row_parts, axis=3))
    patches = np.stack(parts, axis=3)  # H,W,T,P,P,C
    return patches.reshape(height * width, time_steps, patch_size, patch_size, n_channels)
