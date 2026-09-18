import glob, math, sys, os, time, random, argparse
import fiona
import rasterio as rio
import rasterio.mask
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import shape as shapely_shape, mapping, Point
from shapely.ops import unary_union, transform as shapely_transform
try:
    from shapely import make_valid as shapely_make_valid
except ImportError:
    try:
        from shapely.validation import make_valid as shapely_make_valid
    except ImportError:
        shapely_make_valid = None


# Split input vector parcels (train_feature) into training, test and prediction subsets.
# Sample training and test pixels from rasters and save them as CSV tables.
# Save the prediction subset as vector features.
# These outputs support model training, testing and validation.
# Split by parcel to keep pixels from the same parcel together and reduce spatial leakage.

def main(rasters_folder, stacked_raster, train_feature, output_dir, train_ratio, predict_ratio, pixel_ratio, cache,
         inner_buffer_pixels=0, split_lists=None):
    os.environ["GDAL_CACHEMAX"] = cache
    inner_buffer_pixels = float(inner_buffer_pixels or 0)

    # Define predict_shapefile output path at the start
    # Save held-out parcels for post-classification confusion-matrix evaluation in ENVI.
    predict_shapefile = train_feature.replace(".shp", "_predict_samples.shp")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        predict_shapefile = os.path.join(output_dir, "predict_samples.shp")  # Output path.

    test_shapefile = train_feature.replace(".shp", "_test_samples.shp")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        test_shapefile = os.path.join(output_dir, "test_samples.shp")  # Output path.

    train_shapefile = train_feature.replace(".shp", "_train_samples.shp")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        train_shapefile = os.path.join(output_dir, "train_samples.shp")  # Output path.

    # stacking rasters if rasters_folder is provided.
    if rasters_folder:
        # Stacking time series rasters together
        rasters = glob.glob(os.path.join(rasters_folder, "*.tif")) + glob.glob(
            os.path.join(rasters_folder, "*.tiff"))
        stack(rasters, stacked_raster)

    # Generatin train/test datasets
    train_csv = train_feature.replace(".shp", "_train_datasat.csv")
    test_csv = train_feature.replace(".shp", "_test_datasat.csv")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        train_csv = os.path.join(output_dir, os.path.basename(train_csv))
        test_csv = os.path.join(output_dir, os.path.basename(test_csv))

    # Split by vector parcel so training and test pixels come from separate parcels.
    if split_lists is None:
        train_list, test_list, predict_list = split_train_feature(train_feature, train_ratio, predict_ratio)
    else:
        train_list, test_list, predict_list = split_lists

    # Save predict_list as vector features.
    # Vector outputs are saved after CRS normalization and optional inward buffering.

    # Extract raster pixel values within parcels in train_list and test_list.
    # Reproject once here rather than repeatedly inside generate_training_data.
    train_feature_for_extract = prepare_reprojected_shp(stacked_raster, train_feature, output_dir)
    buffer_distance = pixel_buffer_distance(stacked_raster, inner_buffer_pixels)
    save_split_assignments(train_feature_for_extract, train_list, test_list, predict_list, output_dir)
    if inner_buffer_pixels > 0:
        print(
            f"Using inward sample buffer: {inner_buffer_pixels:g} pixel(s), "
            f"distance={buffer_distance:.8f} map units."
        )
    save_predict_as_vector(train_feature_for_extract, predict_list, predict_shapefile, buffer_distance=buffer_distance)
    save_test_as_vector(train_feature_for_extract, test_list, test_shapefile, buffer_distance=buffer_distance)
    save_train_as_vector(train_feature_for_extract, train_list, train_shapefile, buffer_distance=buffer_distance)

    test_combined_csv = generate_training_data(
        stacked_raster, train_feature_for_extract, test_list, test_csv, pixel_ratio,
        inner_buffer_pixels=inner_buffer_pixels, write_distance_metadata=True
    )
    train_combined_csv = generate_training_data(
        stacked_raster, train_feature_for_extract, train_list, train_csv, pixel_ratio,
        inner_buffer_pixels=inner_buffer_pixels, write_distance_metadata=True
    )
    save_master_distance_dataset(train_combined_csv, test_combined_csv, output_dir)


def save_master_distance_dataset(train_csv, test_csv, output_dir):
    """Save one canonical pixel table containing both train and test splits."""
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    train_df.insert(0, "split", "train")
    test_df.insert(0, "split", "test")
    master_df = pd.concat([train_df, test_df], ignore_index=True)
    master_df.insert(0, "sample_id", np.arange(len(master_df), dtype=np.int64))
    master_csv = os.path.join(output_dir, "samples_all_datasat_with_distance.csv")
    master_df.to_csv(master_csv, index=False, float_format="%.6f")
    print(
        f"Canonical distance-weighting dataset saved: {master_csv} "
        f"(rows={len(master_df)}, splits={master_df['split'].value_counts().to_dict()}, "
        f"rings={master_df['ring_id'].value_counts().sort_index().to_dict()})"
    )
    return master_csv


def stack(rasters, out_raster):
    # Read in metadata
    first_raster = rio.open(rasters[0], 'r')
    out_raster = rio.open(out_raster, 'w', nodata=first_raster.nodata, driver='GTiff', height=first_raster.shape[0],
                          width=first_raster.shape[1], count=first_raster.count * len(rasters),
                          dtype=first_raster.read(1).dtype, crs=first_raster.crs, transform=first_raster.transform)
    band_index = 0
    for raster in tqdm(rasters):
        for band in rio.open(raster, "r").read():
            band_index += 1
            out_raster.write(band, band_index)
    out_raster.close()


def interpolation(input_raster, output_raster, n_channels, threshold, cache="50%"):  # Interpolate missing values.
    os.environ["GDAL_CACHEMAX"] = cache
    src = rio.open(input_raster, 'r')
    meta = src.meta.copy()
    nodata = meta["nodata"]
    height = meta["height"]
    width = meta["width"]
    count = meta["count"]
    dtype = meta["dtype"]
    dst = rio.open(output_raster, 'w', nodata=nodata, driver='GTiff',
                   height=height, width=width,
                   count=threshold * n_channels, dtype=dtype,
                   crs=meta["crs"], transform=meta["transform"])
    # Read raster as array [bands, height, width]
    array = src.read()
    # Reshape array
    n_pixels = height * width
    array = array.transpose(1, 2, 0).reshape(n_pixels, int(count / n_channels), n_channels)  # [pixels, time steps, channels]
    # Loop through every pixel
    out_array = np.zeros((n_pixels, threshold, n_channels), dtype=dtype)
    for i, arr in tqdm(enumerate(array), total=n_pixels):
        # Delete nodata sequence
        nodata_seq = np.unique(np.where(arr == nodata)[0])
        arr = np.delete(arr, nodata_seq, axis=0)
        # Define a sequencelength threshold,
        # if sequencelength of pixel >= threshold, randomly extract from sequencelength
        # if sequencelength of 0.5*threshold < pixel < threshold, we randomly interpolate value from sequencelength to it
        # if sequencelength of pixel < 0.5*threshold, turn whole sequence to nodata
        sequencelength = arr.shape[0]
        if sequencelength >= threshold:
            idxs = np.random.choice(sequencelength, threshold, replace=False)
            idxs.sort()
            arr = arr[idxs]
        elif 0.5 * threshold < sequencelength < threshold:
            idxs = np.random.choice(sequencelength, threshold - sequencelength, replace=False)
            idxs = np.append(np.arange(sequencelength), idxs)
            idxs.sort()
            arr = arr[idxs]
        else:
            arr = np.full((threshold, n_channels), nodata)

        # Write in pixel
        out_array[i, :] = arr

    # Reshape to [bands, width, height]
    out_array = out_array.reshape(height, width, -1).transpose(2, 0, 1)
    # Write out_array to out_raster
    dst.write(out_array)


def split_train_feature(train_feature, train_ratio=0.8, predict_ratio=0.1):  # origin: train=0.9 pre=0
    # read train_feature, get classvalue
    with fiona.open(train_feature, "r") as shp:
        classvalues = []
        for fid, feature in enumerate(shp):
            properties = dict(feature["properties"])
            if "class" not in properties:
                raise ValueError(
                    f"Feature {fid} has no 'class' field. "
                    f"Available fields: {list(properties.keys())}"
                )
            value = properties["class"]
            classvalues.append(int(value) if value is not None else 0)
    train_list = []
    test_list = []
    predict_list = []

    for classvalue in list(set(classvalues)):
        fid_list = []
        for i in range(len(classvalues)):
            if classvalues[i] == classvalue:
                fid_list.append(i)

        random.shuffle(fid_list)
        train = fid_list[:int(len(fid_list) * train_ratio)]  # 0.7
        test = fid_list[int(len(fid_list) * train_ratio):int(len(fid_list) * (1 - predict_ratio))]  # 0.7~0.9=0.2
        predict = fid_list[int(len(fid_list) * (1 - predict_ratio)):int(len(fid_list) * 1)]  # 0.9~1.0=0.1
        [train_list.append(i) for i in train]
        [test_list.append(i) for i in list(test)]
        [predict_list.append(i) for i in list(predict)]
    return train_list, test_list, predict_list


def prepare_reprojected_shp(input_raster, train_feature, output_dir):
    with rio.open(input_raster) as src:
        raster_crs = src.crs

    # Fiona/OGR can recover unclosed rings that pyogrio rejects before Shapely
    # gets a chance to repair them. Keep the source row order unchanged because
    # the train/test split and exported polygon_id use this positional FID.
    gdf = gpd.read_file(train_feature, engine="fiona")
    repaired_count = 0
    empty_fids = []
    repaired_geometries = []
    for fid, geometry in enumerate(gdf.geometry):
        if geometry is None or geometry.is_empty:
            repaired_geometries.append(None)
            empty_fids.append(fid)
            continue
        if geometry.is_valid:
            repaired_geometries.append(geometry)
            continue

        repaired = shapely_make_valid(geometry) if shapely_make_valid is not None else geometry.buffer(0)
        # make_valid may return a GeometryCollection containing lines/points.
        # Sampling only supports polygonal parts, so discard non-area pieces.
        if repaired.geom_type == "GeometryCollection":
            polygon_parts = [
                part for part in repaired.geoms
                if part.geom_type in ("Polygon", "MultiPolygon") and not part.is_empty
            ]
            repaired = unary_union(polygon_parts) if polygon_parts else None
        if repaired is None or repaired.is_empty:
            repaired_geometries.append(None)
            empty_fids.append(fid)
            continue
        repaired_geometries.append(repaired)
        repaired_count += 1

    if empty_fids:
        raise ValueError(
            "Geometry repair produced empty/non-polygon features at positional FIDs: "
            + ", ".join(map(str, empty_fids[:20]))
        )
    gdf = gdf.set_geometry(repaired_geometries)

    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    # Always write the normalized copy when geometry was repaired or CRS was
    # transformed; main() then uses exactly this copy for vectors and CSVs.
    os.makedirs(output_dir, exist_ok=True)
    out_shp = os.path.join(output_dir, "sample_shape_reprojected_extract.shp")
    base = os.path.splitext(out_shp)[0]
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj"]:
        f = base + ext
        if os.path.exists(f):
            os.remove(f)

    gdf.to_file(out_shp, driver="ESRI Shapefile", encoding="utf-8")
    print(
        f"Geometry normalization complete: repaired={repaired_count}, "
        f"features={len(gdf)}, output={out_shp}"
    )
    return out_shp


def pixel_buffer_distance(raster_path, inner_buffer_pixels):
    if inner_buffer_pixels is None or float(inner_buffer_pixels) <= 0:
        return 0.0
    with rio.open(raster_path) as src:
        transform = src.transform
        pixel_width = math.sqrt(transform.a ** 2 + transform.d ** 2)
        pixel_height = math.sqrt(transform.b ** 2 + transform.e ** 2)
        pixel_size = max(abs(pixel_width), abs(pixel_height))
    return float(inner_buffer_pixels) * pixel_size


def inner_buffer_geometry(geometry, buffer_distance):
    if not geometry:
        return geometry
    geom = shapely_shape(geometry)
    # Rasterio 1.x in the bundled QGIS Python environment cannot apply a 2-D
    # affine transform to (x, y, z) coordinate tuples.  Sampling is planar, so
    # explicitly discard Z before masking/buffering.
    if getattr(geom, "has_z", False):
        geom = shapely_transform(lambda x, y, z=None: (x, y), geom)
    # Older Rasterio versions bundled with QGIS may not accept Fiona's
    # Geometry proxy directly.  Always return a plain GeoJSON mapping.
    if buffer_distance <= 0:
        return mapping(geom)
    buffered = geom.buffer(-float(buffer_distance))
    if buffered.is_empty:
        return None
    if buffered.geom_type == "GeometryCollection":
        polygons = [
            part for part in buffered.geoms
            if part.geom_type in ("Polygon", "MultiPolygon") and not part.is_empty
        ]
        if not polygons:
            return None
        buffered = unary_union(polygons)
    if buffered.is_empty:
        return None
    return mapping(buffered)


def parse_buffer_pixels(value):
    if value is None:
        return [0.0]
    values = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values or [0.0]


def buffer_output_dir(base_output_dir, buffer_pixels, use_subdir):
    if not use_subdir:
        return base_output_dir
    label = ("%g" % float(buffer_pixels)).replace(".", "p")
    return os.path.join(base_output_dir, f"buffer_px_{label}")


def save_split_assignments(train_feature, train_list, test_list, predict_list, output_dir):
    if not output_dir:
        return
    split_by_fid = {}
    for fid in train_list:
        split_by_fid[int(fid)] = "train"
    for fid in test_list:
        split_by_fid[int(fid)] = "test"
    for fid in predict_list:
        split_by_fid[int(fid)] = "predict"

    rows = []
    with fiona.open(train_feature, "r") as src:
        for fid, feature in enumerate(src):
            if fid not in split_by_fid:
                continue
            rows.append({
                "fid": int(fid),
                "split": split_by_fid[fid],
                "class": int(feature["properties"]["class"]) if feature["properties"]["class"] is not None else 0,
            })
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "split_assignments.csv"), index=False)

# Extract samples from input_raster using vector features in train_feature.
# Save the resulting pixel-value dataset to out_csv.
def generate_training_data(input_raster, train_feature, sample_list, out_csv, pixel_ratio=0.8,
                           inner_buffer_pixels=0, write_distance_metadata=False):
    # read input_raster and train_feature
    src = rio.open(input_raster)  # Open the raster with Rasterio to access its metadata.
    shp = fiona.open(train_feature, "r")
    buffer_distance = pixel_buffer_distance(input_raster, inner_buffer_pixels)
    if not 0 < pixel_ratio <= 1:
        raise ValueError(f"pixel_ratio must be in (0, 1], got {pixel_ratio}")

    # generate empty array to store data
    # Allocate out_images to store sampled data.
    # Reserve rows based on raster size and columns for features and labels.
    out_images = np.zeros(shape=(int(src.width * src.height / 3), src.count + 2), dtype=np.float32)
    print(src)
    print(f"Preallocated sample array: shape={out_images.shape}, dtype={out_images.dtype}")

    row0 = 0  # Track the next output row.
    masked_pixel_count = 0
    nonfinite_pixel_count = 0
    empty_geometry_count = 0
    metadata_rows = []
    pixel_size = pixel_buffer_distance(input_raster, 1)

    # loop through every feature(polygon) to extract pixels' value within it.
    # Iterate over polygon features with a tqdm progress bar.
    with tqdm(sample_list, desc="Generating test/train datasets") as samples:
        for i in samples:
            # Masking raster using feature
            geometry = inner_buffer_geometry(shp[i]["geometry"], buffer_distance)
            if geometry is None:
                empty_geometry_count += 1
                continue
            shape = [geometry]
            # Preserve the boolean mask: without nodata metadata, filled=True may fill
            # pixels outside the polygon with zero and incorrectly include them as samples.
            masked_image, masked_transform = rio.mask.mask(
                src,
                shape,
                crop=True,
                filled=False,
            )

            # Reshape [bands, rows, cols] to [pixels, bands]. Exclude pixels masked
            # in any band, removing both outside-polygon and nodata pixels.
            out_image = masked_image.data.transpose(1, 2, 0).reshape(-1, src.count)
            pixel_mask = np.ma.getmaskarray(masked_image).transpose(1, 2, 0).reshape(-1, src.count)
            valid_mask = ~pixel_mask.any(axis=1)
            masked_pixel_count += int((~valid_mask).sum())
            out_image = out_image[valid_mask]

            # Remove NaN/Inf explicitly because nodata metadata may not identify them.
            finite_mask = np.isfinite(out_image).all(axis=1)
            nonfinite_pixel_count += int((~finite_mask).sum())
            out_image = out_image[finite_mask]
            if out_image.shape[0] == 0:
                continue

            # For the unbuffered master dataset, write row-aligned spatial
            # metadata to a companion CSV.  The feature CSV remains unchanged
            # (class, polygon_id, raster bands), so existing training code does
            # not accidentally treat distance/ring fields as model features.
            if write_distance_metadata:
                original_geometry = shapely_shape(shp[i]["geometry"])
                boundary = original_geometry.boundary
                crop_rows, crop_cols = np.indices(masked_image.shape[1:])
                flat_rows = crop_rows.reshape(-1)[valid_mask][finite_mask]
                flat_cols = crop_cols.reshape(-1)[valid_mask][finite_mask]
                xs, ys = rio.transform.xy(masked_transform, flat_rows, flat_cols, offset="center")
                for x, y in zip(xs, ys):
                    distance_map = float(boundary.distance(Point(float(x), float(y))))
                    distance_px = distance_map / pixel_size if pixel_size > 0 else 0.0
                    # ring 0: [0, 1 px), ring 1: [1, 2 px), ring 2: >= 2 px.
                    # This exactly supports the planned (w0, w1, w2) schemes.
                    ring_id = min(2, max(0, int(math.floor(distance_px + 1e-9))))
                    global_row, global_col = src.index(float(x), float(y))
                    metadata_rows.append({
                        "polygon_id": int(i),
                        "ring_id": int(ring_id),
                        "distance_px": float(distance_px),
                        "distance_map": float(distance_map),
                        "raster_row": int(global_row),
                        "raster_col": int(global_col),
                        "x": float(x),
                        "y": float(y),
                    })

            # Add classvalue to every row
            classvalue = int(shp[i]['properties']['class'])  # class1
            # classvalue = int(shp[i]['properties']['Classvalue'])
            lc_id = np.full((out_image.shape[0], 1), classvalue, dtype=np.float32)
            polygon_id = np.full((out_image.shape[0], 1), int(i), dtype=np.float32)
            out_image = np.concatenate((lc_id, polygon_id, out_image), axis=1)
            # Write in out_image
            row1 = row0 + out_image.shape[0]
            if row1 > out_images.shape[0]:
                new_size = max(row1, int(out_images.shape[0] * 1.5))
                expanded = np.zeros((new_size, src.count + 2), dtype=np.float32)
                expanded[:row0] = out_images[:row0]
                out_images = expanded
            out_images[row0:row1, :] = out_image
            row0 += out_image.shape[0]

    # Delete all-zero rows from out_images.
    out_images = out_images[:row0]
    print(
        "Pixel filtering summary: "
        f"outside/nodata={masked_pixel_count}, "
        f"NaN/Inf={nonfinite_pixel_count}, "
        f"empty buffered geometries={empty_geometry_count}, "
        f"valid={row0}"
    )
    if out_images.shape[0] == 0:
        shp.close()
        src.close()
        raise ValueError(f"No valid pixels were extracted from {len(sample_list)} features")

    # To avoid similarity, only part of the whole data are exported
    # Indices's datatype should be int32 cause maximum value of int16 is 32767
    if pixel_ratio < 1:
        indices = np.arange(0, out_images.shape[0], 1 / pixel_ratio).astype(np.int64)
        out_images = out_images[indices]
        if write_distance_metadata:
            metadata_rows = [metadata_rows[int(idx)] for idx in indices]
    # The second column is the source polygon id. Downstream parcel-aware
    # validation and parcel-consistency loss rely on this id being stable.
    # Export to .csv format
    print("================================")

    print("dtype =", out_images.dtype)

    print(
        "min =",
        np.nanmin(out_images[:, 2:])
    )

    print(
        "max =",
        np.nanmax(out_images[:, 2:])
    )

    print(
        "mean =",
        np.nanmean(out_images[:, 2:])
    )

    print("================================")
    df = pd.DataFrame(out_images)
    df.to_csv(
        out_csv,
        index=False,
        header=False,
        float_format="%.6f"
    )
    if write_distance_metadata:
        metadata_csv = os.path.splitext(out_csv)[0] + "_pixel_metadata.csv"
        metadata_df = pd.DataFrame(metadata_rows)
        if len(metadata_df) != len(df):
            raise RuntimeError(
                f"Feature/metadata row mismatch: features={len(df)}, metadata={len(metadata_df)}"
            )
        metadata_df.insert(0, "row_id", np.arange(len(metadata_df), dtype=np.int64))
        metadata_df.to_csv(metadata_csv, index=False, float_format="%.6f")

        # Also emit a human-readable combined table.  The original headerless
        # feature CSV is deliberately retained for backward compatibility.
        feature_columns = ["class", "polygon_id"] + [
            f"feature_{idx}" for idx in range(src.count)
        ]
        combined_df = df.copy()
        combined_df.columns = feature_columns
        spatial_columns = [
            "ring_id", "distance_px", "distance_map",
            "raster_row", "raster_col", "x", "y",
        ]
        for column in spatial_columns:
            combined_df[column] = metadata_df[column].to_numpy()
        combined_csv = os.path.splitext(out_csv)[0] + "_with_distance.csv"
        combined_df.to_csv(combined_csv, index=False, float_format="%.6f")
        print(
            f"Distance metadata saved: {metadata_csv} "
            f"(rows={len(metadata_df)}, ring_counts="
            f"{metadata_df['ring_id'].value_counts().sort_index().to_dict()})"
        )
        print(f"Combined feature + distance dataset saved: {combined_csv}")
        result_csv = combined_csv
    else:
        result_csv = out_csv
    shp.close()
    src.close()
    return result_csv
    # print(src.dtypes)


def save_predict_as_vector(train_feature, predict_list, output_shp, buffer_distance=0.0):
    # Read the original vector data.
    with fiona.open(train_feature, "r") as src:
        # Define the output shapefile schema.
        schema = src.schema.copy()
        crs = src.crs

        # Create the output shapefile.
        with fiona.open(output_shp, "w", driver="ESRI Shapefile", schema=schema, crs=crs) as dst:
            for i in predict_list:
                # Copy and write the selected samples.
                feature = dict(src[i])
                geometry = inner_buffer_geometry(feature.get("geometry"), buffer_distance)
                if geometry is None:
                    continue
                feature["geometry"] = geometry
                dst.write(feature)


def save_test_as_vector(train_feature, test_list, output_shp, buffer_distance=0.0):
    # Read the original vector data.
    with fiona.open(train_feature, "r") as src:
        # Define the output shapefile schema.
        schema = src.schema.copy()
        crs = src.crs

        # Create the output shapefile.
        with fiona.open(output_shp, "w", driver="ESRI Shapefile", schema=schema, crs=crs) as dst:
            for i in test_list:
                # Copy and write the selected samples.
                feature = dict(src[i])
                geometry = inner_buffer_geometry(feature.get("geometry"), buffer_distance)
                if geometry is None:
                    continue
                feature["geometry"] = geometry
                dst.write(feature)


def save_train_as_vector(train_feature, train_list, output_shp, buffer_distance=0.0):
    # Read the original vector data.
    with fiona.open(train_feature, "r") as src:
        # Define the output shapefile schema.
        schema = src.schema.copy()
        crs = src.crs

        # Create the output shapefile.
        with fiona.open(output_shp, "w", driver="ESRI Shapefile", schema=schema, crs=crs) as dst:
            for i in train_list:
                # Copy and write the selected samples.
                feature = dict(src[i])
                geometry = inner_buffer_geometry(feature.get("geometry"), buffer_distance)
                if geometry is None:
                    continue
                feature["geometry"] = geometry
                dst.write(feature)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generating training datasets')
    parser.add_argument('--rasters_folder', dest='rasters_folder',
                        help='rasters folder',
                        default=None)
    parser.add_argument('--stacked_raster', dest='stacked_raster',
                        help='path for stacked raster',
                        required=True)
    parser.add_argument('--train_feature', dest='train_feature',
                        help='path for train feature',
                        required=True)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='path to store datasets',
                        required=True)
    parser.add_argument('--train_ratio', dest='train_ratio', type=float,
                        help='ratio of train part', default=0.7)
    parser.add_argument('--predict_ratio', dest='predict_ratio', type=float,
                        help='ratio of predict part, for TF/Keras only', default=0.0)  # 0.1
    parser.add_argument('--pixel_ratio', dest='pixel_ratio', type=float,  # Fraction of pixels sampled from the input samples.
                        help='ratio of extracted pixel values to keep', default=1.0)
    parser.add_argument('--split_seed', dest='split_seed', type=int,
                        help='random seed for a shared train/test/predict split across buffer variants',
                        default=42)
    parser.add_argument('--cache', dest='cache',
                        help='Set GDAL raster block cache size, may speed up processing with higher percentage, default is 5%% of usable physical RAM',
                        type=str, default='20%')
    args = parser.parse_args()

    random.seed(args.split_seed)
    shared_split_lists = split_train_feature(args.train_feature, args.train_ratio, args.predict_ratio)
    print(f"Shared parcel split generated with seed={args.split_seed}.")
    print("Making one unbuffered master dataset with per-pixel boundary distance attributes.")
    main(
        args.rasters_folder, args.stacked_raster, args.train_feature, args.output_dir,
        args.train_ratio, args.predict_ratio, args.pixel_ratio, args.cache,
        inner_buffer_pixels=0, split_lists=shared_split_lists
    )
