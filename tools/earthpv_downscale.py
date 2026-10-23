#!/usr/bin/env python3
"""Downscale an EarthPV per-cell capacity table to a fine point grid for posters.

EarthPV publishes its solar-PV capacity estimate per 0.1° cell (~11 km), e.g.
``pakistan_evidence_atlas_capacity_by_cell.parquet`` with the Best estimate in
``mwp_best``. At poster scale that reads as a soft heat map. This tool spreads
each cell's published total over the cell using finer per-building or
per-detection layers as *weights only* (dasymetric downscaling), so:

- every cell keeps exactly its published capacity (totals are preserved),
- the within-cell texture follows where the model actually found PV.

Cells whose weights sum to zero are spread evenly over a sub-grid of the cell.

Weights only need to be roughly proportional to capacity, so area layers are
scaled to kWp: raw segmentation candidates at ~0.03 kWp/m² (their published
recall-corrected yield; many are unconfirmed) and OSM-mapped arrays at
~0.05 kWp/m² (EarthPV's OSM MWp over mapped area).

This is the command that built ``data/earthpv/pakistan_pv_best_1km.parquet``,
run from an EarthPV checkout::

    python tools/earthpv_downscale.py \\
        --cells docs/assets/interactive/pakistan_evidence_atlas_capacity_by_cell.parquet \\
        --weights data/roofclf_national_with_sppi/pakistan/density/sub400_central_incremental_buildings.parquet:est_kwp_sub400 \\
        --weights data/roofclf_national_with_sppi/pakistan/density/ge400_roof_incremental_buildings.parquet:est_kwp_ge400_roof \\
        --weights data/predictions/pakistan/candidates.parquet:area_m2:0.03 \\
        --weights data/labels/pakistan_overpass_solar.parquet:area_m2:0.05 \\
        --out ../grid2poster/data/earthpv/pakistan_pv_best_1km.parquet

The output is a plain parquet with ``lon``, ``lat`` (WGS84 bin centres) and
``mwp`` columns, for ``create_grid_poster.py --earthpv PATH``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parse_weight_spec(spec: str) -> tuple[Path, str, float]:
    """Parse ``PATH:COLUMN[:FACTOR]`` into its parts."""
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(f"expected PATH:COLUMN[:FACTOR], got {spec!r}")
    factor = float(parts[2]) if len(parts) == 3 else 1.0
    return Path(parts[0]), parts[1], factor


def load_weight_points(path: Path, column: str, factor: float) -> pd.DataFrame:
    frame = gpd.read_parquet(path, columns=["geometry", column]).to_crs("EPSG:4326")
    frame = frame[frame[column].fillna(0) > 0]
    # Representative points stay inside footprints; centroids of thin or
    # concave polygons can fall outside them (and into a neighbouring cell).
    points = frame.geometry.representative_point()
    weights = frame[column].to_numpy(dtype="float64") * factor
    print(f"  {path.name}: {len(frame):,} features, Σ {column}×{factor:g} = {weights.sum():,.0f}")
    return pd.DataFrame({"lon": points.x.to_numpy(), "lat": points.y.to_numpy(), "w": weights})


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=Path, required=True, help="EarthPV per-cell capacity parquet")
    parser.add_argument("--capacity-column", default="mwp_best", help="Cell capacity column in MWp")
    parser.add_argument(
        "--weights",
        type=parse_weight_spec,
        action="append",
        default=[],
        metavar="PATH:COLUMN[:FACTOR]",
        help="A layer whose COLUMN (times FACTOR) sets the within-cell distribution. Repeatable.",
    )
    parser.add_argument("--cell-size", type=float, default=0.1, help="Cell size in degrees")
    parser.add_argument("--resolution", type=float, default=0.01, help="Output bin size in degrees")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    cells = gpd.read_parquet(args.cells)
    cells = cells[cells[args.capacity_column].fillna(0) > 0]
    total = float(cells[args.capacity_column].sum())
    print(f"{len(cells):,} cells with capacity, Σ {args.capacity_column} = {total:,.1f} MWp")

    # Cells are keyed by their lower-left corner; snap everything onto the same
    # lattice so points and cells agree on membership.
    step = args.cell_size
    origin_lon = float(cells["lon0"].min())
    origin_lat = float(cells["lat0"].min())

    def cell_index(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        ix = np.floor((lon - origin_lon) / step + 1e-9).astype(np.int64)
        iy = np.floor((lat - origin_lat) / step + 1e-9).astype(np.int64)
        return ix * 100_000 + iy

    cells = cells.assign(key=cell_index(cells["lon0"].to_numpy() + step / 2, cells["lat0"].to_numpy() + step / 2))
    capacity = cells.set_index("key")[args.capacity_column]

    print("Loading weight layers:")
    weights = pd.concat([load_weight_points(*spec) for spec in args.weights], ignore_index=True) if args.weights else pd.DataFrame(columns=["lon", "lat", "w"])
    weights["key"] = cell_index(weights["lon"].to_numpy(), weights["lat"].to_numpy())
    weights = weights[weights["key"].isin(capacity.index)]
    cell_weight = weights.groupby("key")["w"].sum()
    weights["mwp"] = weights["w"] / weights["key"].map(cell_weight) * weights["key"].map(capacity)

    # Cells without any weight: spread their capacity evenly over a sub-grid.
    empty = capacity[~capacity.index.isin(cell_weight.index)]
    sub = max(int(round(step / args.resolution)), 1)
    offsets = (np.arange(sub) + 0.5) / sub * step
    fill = []
    if not empty.empty:
        lon0 = cells.set_index("key").loc[empty.index, "lon0"].to_numpy()
        lat0 = cells.set_index("key").loc[empty.index, "lat0"].to_numpy()
        gx, gy = np.meshgrid(offsets, offsets)
        fill = pd.DataFrame({
            "lon": (lon0[:, None] + gx.ravel()[None, :]).ravel(),
            "lat": (lat0[:, None] + gy.ravel()[None, :]).ravel(),
            "mwp": np.repeat(empty.to_numpy() / gx.size, gx.size),
        })
        print(f"{len(empty):,} cells had no weights; spread {empty.sum():,.1f} MWp evenly")

    points = pd.concat([weights[["lon", "lat", "mwp"]], *([fill] if len(fill) else [])], ignore_index=True)

    # Bin to the output resolution on the same lattice as the cells.
    res = args.resolution
    bx = np.floor((points["lon"].to_numpy() - origin_lon) / res + 1e-9).astype(np.int64)
    by = np.floor((points["lat"].to_numpy() - origin_lat) / res + 1e-9).astype(np.int64)
    binned = (
        pd.DataFrame({"bx": bx, "by": by, "mwp": points["mwp"].to_numpy()})
        .groupby(["bx", "by"], sort=True)["mwp"].sum()
        .reset_index()
    )
    out = pd.DataFrame({
        "lon": (origin_lon + (binned["bx"] + 0.5) * res).astype("float32"),
        "lat": (origin_lat + (binned["by"] + 0.5) * res).astype("float32"),
        "mwp": binned["mwp"].astype("float32"),
    })
    out = out[out["mwp"] > 0]

    written = float(out["mwp"].astype("float64").sum())
    print(f"{len(out):,} bins at {res:g}°, Σ mwp = {written:,.1f} MWp (cells: {total:,.1f})")
    if abs(written - total) > 1e-3 * max(total, 1.0):
        print("Error: binned total does not match the cell total", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": args.cells.name,
        "capacity_column": args.capacity_column,
        "total_mwp": round(total, 3),
        "resolution_deg": res,
        "weights": [f"{p.name}:{c}:{f:g}" for p, c, f in args.weights],
    }
    table = pa.Table.from_pandas(out, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"earthpv_downscale": json.dumps(meta).encode()})
    pq.write_table(table, args.out, compression="zstd")
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
