"""Solar PV capacity layers (EarthPV atlases or any capacity file) for the glow overlay."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.ops import unary_union

from common import CACHE_DIR, cache_key, slugify

# EarthPV publishes one capacity-by-cell table per country next to its atlas
# pages: https://open-energy-transition.github.io/earthpv/
EARTHPV_BASE_URL = "https://open-energy-transition.github.io/earthpv/assets/interactive/"
# Pakistan, the first atlas, predates the "<country>_pv_evidence_atlas" naming.
EARTHPV_FILE_OVERRIDES: dict[str, str] = {
    "pakistan": "pakistan_evidence_atlas_capacity_by_cell.parquet",
}

# Capacity columns (MWp) recognised when --pv-column is not given, in order of
# preference. "mwp_best" is the EarthPV atlas's headline Best estimate.
PV_CAPACITY_COLUMNS: tuple[str, ...] = ("mwp", "mwp_best", "capacity_mwp", "capacity_mw", "mw")

# Points within this distance of the boundary are kept, so capacity binned on a
# slightly different border dataset is not lost along the edge.
PV_BOUNDARY_TOLERANCE_M = 5000.0

# Polygon capacity (e.g. EarthPV's 0.1° cells) is spread over a lattice whose
# spacing is this fraction of the boundary's larger side - about one glow
# raster pixel on an A3 poster - so cells read as an area, not a dot.
PV_SPREAD_SPACING_FRACTION = 1 / 2000
PV_SPREAD_MAX_POINTS = 2500


def earthpv_url(country: str) -> str:
    slug = slugify(country)
    filename = EARTHPV_FILE_OVERRIDES.get(slug, f"{slug}_pv_evidence_atlas_capacity_by_cell.parquet")
    return EARTHPV_BASE_URL + filename


def _download(url: str, use_cache: bool) -> Path:
    suffix = Path(url.split("?")[0]).suffix or ".bin"
    path = CACHE_DIR / f"pv_{cache_key('pv_layer_v1', url)}{suffix}"
    if use_cache and path.exists():
        print(f"Using cached PV layer {url}")
        return path
    print(f"Downloading PV layer {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Could not download PV layer {url} (HTTP {exc.code})") from exc
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


def _read_table(path: Path) -> gpd.GeoDataFrame:
    """Read a GeoParquet/GeoJSON/GPKG file, or a parquet/CSV with lon/lat columns."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            return gpd.read_parquet(path)
        except ValueError:
            # Plain parquet without GeoParquet metadata: fall through to lon/lat.
            frame = pd.read_parquet(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        return gpd.read_file(path)

    columns = {c.lower(): c for c in frame.columns}
    lon = next((columns[c] for c in ("lon", "longitude", "x") if c in columns), None)
    lat = next((columns[c] for c in ("lat", "latitude", "y") if c in columns), None)
    if lon is None or lat is None:
        raise SystemExit(f"PV layer {path} has no geometry and no lon/lat columns")
    return gpd.GeoDataFrame(frame, geometry=gpd.points_from_xy(frame[lon], frame[lat]), crs="EPSG:4326")


def load_pv_layer(source: str, column: str | None = None, use_cache: bool = True) -> gpd.GeoDataFrame:
    """Load a PV capacity layer as WGS84 features with an ``mwp`` column.

    ``source`` is a local path or an http(s) URL. Geometries may be points or
    polygons; prepare_pv() spreads polygon capacity over the polygon area.
    """
    path = _download(source, use_cache) if source.startswith(("http://", "https://")) else Path(source)
    if not path.exists():
        raise SystemExit(f"PV layer not found: {path}")
    layer = _read_table(path)
    if layer.crs is None:
        layer = layer.set_crs("EPSG:4326")

    if column is None:
        column = next((c for c in PV_CAPACITY_COLUMNS if c in layer.columns), None)
        if column is None:
            raise SystemExit(
                f"PV layer {path} has none of the capacity columns {', '.join(PV_CAPACITY_COLUMNS)}; "
                "name one with --pv-column"
            )
    elif column not in layer.columns:
        raise SystemExit(f"PV layer {path} has no column '{column}'")

    mwp = pd.to_numeric(layer[column], errors="coerce").fillna(0.0)
    layer = gpd.GeoDataFrame({"mwp": mwp.to_numpy(dtype="float64")}, geometry=layer.geometry.values, crs=layer.crs)
    layer = layer[(layer["mwp"] > 0) & ~layer.geometry.is_empty].to_crs("EPSG:4326")
    print(f"PV layer: {len(layer):,} features, {layer['mwp'].sum():,.0f} MWp ({column})")
    return layer


def _spread_polygons(polygons: gpd.GeoDataFrame, spacing: float) -> gpd.GeoDataFrame:
    """Split each polygon's capacity evenly over lattice points inside it.

    Polygons smaller than one lattice step collapse to their representative
    point, so building-level inputs stay as cheap as plain points.
    """
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    for geom, mwp in zip(polygons.geometry.values, polygons["mwp"].to_numpy()):
        minx, miny, maxx, maxy = geom.bounds
        step = spacing
        if (maxx - minx) * (maxy - miny) / step**2 > PV_SPREAD_MAX_POINTS:
            step = np.sqrt((maxx - minx) * (maxy - miny) / PV_SPREAD_MAX_POINTS)
        nx = int((maxx - minx) / step)
        ny = int((maxy - miny) / step)
        if nx >= 2 or ny >= 2:
            gx, gy = np.meshgrid(
                minx + (np.arange(max(nx, 1)) + 0.5) * (maxx - minx) / max(nx, 1),
                miny + (np.arange(max(ny, 1)) + 0.5) * (maxy - miny) / max(ny, 1),
            )
            gx, gy = gx.ravel(), gy.ravel()
            inside = shapely.contains_xy(geom, gx, gy)
            if inside.any():
                xs.append(gx[inside])
                ys.append(gy[inside])
                ws.append(np.full(int(inside.sum()), mwp / inside.sum()))
                continue
        point = geom.representative_point()
        xs.append(np.array([point.x]))
        ys.append(np.array([point.y]))
        ws.append(np.array([mwp]))
    return gpd.GeoDataFrame(
        {"mwp": np.concatenate(ws)},
        geometry=gpd.points_from_xy(np.concatenate(xs), np.concatenate(ys)),
        crs=polygons.crs,
    )


def prepare_pv(pv: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame, output_crs: str) -> gpd.GeoDataFrame:
    """Project PV features to points and keep those inside (or just outside) the boundary."""
    if pv.empty:
        return pv
    projected = pv.to_crs(output_crs)
    boundary_projected = boundary.to_crs(output_crs)

    is_area = projected.geometry.type.isin(["Polygon", "MultiPolygon"]).to_numpy()
    if is_area.any():
        minx, miny, maxx, maxy = boundary_projected.total_bounds
        spacing = max(maxx - minx, maxy - miny) * PV_SPREAD_SPACING_FRACTION
        spread = _spread_polygons(projected[is_area], spacing)
        points = projected[~is_area].assign(geometry=projected[~is_area].geometry.representative_point())
        projected = gpd.GeoDataFrame(pd.concat([points, spread], ignore_index=True), crs=output_crs)

    mask_geom = unary_union(boundary_projected.geometry).buffer(PV_BOUNDARY_TOLERANCE_M)
    shapely.prepare(mask_geom)
    inside = shapely.contains(mask_geom, projected.geometry.values)
    dropped = float(projected.loc[~inside, "mwp"].sum())
    if dropped >= 0.5:
        print(f"PV layer: {dropped:,.0f} MWp lies outside the boundary and is not drawn")
    return projected[inside]
