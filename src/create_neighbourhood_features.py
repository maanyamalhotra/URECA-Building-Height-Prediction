import math
import os
import re
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from pyproj import Transformer

EARTH_RADIUS_M = 6_371_000.0
RADIUS_M = 250.0
K_NEIGHBOURS = 10

FEATURES_CSV = "features.csv"
LABELS_CSV = "labels_real_zoning_10378.csv"
URA_ZONES_CSV = "building_zones_ura_FIXED.csv"
COORDS_LATLON_CSV = "coordinates_latlon_CORRECT.csv"
COORDS_SVY21_CSV = "coordinates.csv"


def clean_zone_name(z):
    z = str(z).upper().strip()
    z = re.sub(r"[^A-Z0-9]+", "_", z)
    z = z.strip("_")
    return z if z else "UNKNOWN"


def load_coordinates():
    if os.path.exists(COORDS_SVY21_CSV):
        coords = pd.read_csv(COORDS_SVY21_CSV)

        transformer = Transformer.from_crs(
            "EPSG:3414",
            "EPSG:4326",
            always_xy=True
        )

        lon, lat = transformer.transform(
            coords["x"].values,
            coords["y"].values
        )

        coords["longitude"] = lon
        coords["latitude"] = lat

        coords[["building_id", "latitude", "longitude"]].to_csv(
            COORDS_LATLON_CSV,
            index=False
        )

        return coords[["building_id", "latitude", "longitude"]]

    if os.path.exists(COORDS_LATLON_CSV):
        return pd.read_csv(COORDS_LATLON_CSV)[
            ["building_id", "latitude", "longitude"]
        ]

    raise FileNotFoundError("No coordinates file found.")


def safe_mean(values, default=0.0):
    return float(np.mean(values)) if len(values) else default


def safe_std(values, default=0.0):
    return float(np.std(values)) if len(values) else default


def safe_min(values, default=-1.0):
    return float(np.min(values)) if len(values) else default


def main():
    features = pd.read_csv(FEATURES_CSV)
    labels = pd.read_csv(LABELS_CSV)
    zones = pd.read_csv(URA_ZONES_CSV)
    coords = load_coordinates()

    zones["ura_zone"] = zones["ura_zone"].fillna("Unmatched")
    zones["ura_zone_clean"] = zones["ura_zone"].apply(clean_zone_name)

    df = labels.merge(
        features,
        on="building_id",
        how="inner",
        suffixes=("_labels", "_features")
    )

    df = df.merge(
        zones[["building_id", "ura_zone", "ura_zone_clean"]],
        on="building_id",
        how="left"
    )

    df = df.merge(coords, on="building_id", how="inner")

    df["ura_zone"] = df["ura_zone"].fillna("Unmatched")
    df["ura_zone_clean"] = df["ura_zone_clean"].fillna("UNMATCHED")

    required_geom = ["area", "perimeter", "bbox_w", "bbox_h", "aspect_ratio"]
    missing = [c for c in required_geom if c not in df.columns]

    if missing:
        raise ValueError(f"Missing geometry columns: {missing}")

    print(f"Merged dataset: {len(df)} buildings")
    print("\nURA zone distribution:")
    print(df["ura_zone"].value_counts().head(20))

    latlon_rad = np.radians(df[["latitude", "longitude"]].values)
    tree = BallTree(latlon_rad, metric="haversine")

    radius_rad = RADIUS_M / EARTH_RADIUS_M

    radius_indices, radius_distances = tree.query_radius(
        latlon_rad,
        r=radius_rad,
        return_distance=True,
        sort_results=True
    )

    k_query = min(K_NEIGHBOURS + 1, len(df))
    knn_distances_rad, knn_indices = tree.query(latlon_rad, k=k_query)

    areas = df["area"].values.astype(float)
    perimeters = df["perimeter"].values.astype(float)
    bbox_w = df["bbox_w"].values.astype(float)
    bbox_h = df["bbox_h"].values.astype(float)
    aspect = df["aspect_ratio"].values.astype(float)
    ura_zones = df["ura_zone_clean"].values

    unique_zones = sorted(df["ura_zone_clean"].unique())
    print(f"\nNumber of URA zone categories: {len(unique_zones)}")
    print(unique_zones)

    nearest_zone_dist = {}

    for zone in unique_zones:
        mask = ura_zones == zone

        if mask.sum() == 0:
            nearest_zone_dist[zone] = np.full(len(df), -1.0)
            continue

        zone_tree = BallTree(latlon_rad[mask], metric="haversine")
        d_rad, _ = zone_tree.query(latlon_rad, k=1)
        nearest_zone_dist[zone] = d_rad[:, 0] * EARTH_RADIUS_M

    records = []
    radius_area_m2 = math.pi * (RADIUS_M ** 2)

    for i, building_id in enumerate(df["building_id"].values):
        r_idx = radius_indices[i]
        r_dist_m = radius_distances[i] * EARTH_RADIUS_M

        keep = r_idx != i
        r_idx = r_idx[keep]
        r_dist_m = r_dist_m[keep]

        k_idx = knn_indices[i]
        k_dist_m = knn_distances_rad[i] * EARTH_RADIUS_M

        keep = k_idx != i
        k_idx = k_idx[keep][:K_NEIGHBOURS]
        k_dist_m = k_dist_m[keep][:K_NEIGHBOURS]

        neighbour_zones = ura_zones[r_idx]
        current_zone = ura_zones[i]

        same_zone_ratio = (
            float(np.mean(neighbour_zones == current_zone))
            if len(neighbour_zones)
            else 0.0
        )

        if len(neighbour_zones):
            vals, counts = np.unique(neighbour_zones, return_counts=True)
            dominant_zone = vals[np.argmax(counts)]
        else:
            dominant_zone = "NONE"

        rec = {
            "building_id": building_id,

            f"radius_{int(RADIUS_M)}m_count": int(len(r_idx)),
            f"radius_{int(RADIUS_M)}m_density": float(len(r_idx) / radius_area_m2),

            f"radius_{int(RADIUS_M)}m_mean_area": safe_mean(areas[r_idx]),
            f"radius_{int(RADIUS_M)}m_std_area": safe_std(areas[r_idx]),
            f"radius_{int(RADIUS_M)}m_mean_perimeter": safe_mean(perimeters[r_idx]),
            f"radius_{int(RADIUS_M)}m_mean_bbox_w": safe_mean(bbox_w[r_idx]),
            f"radius_{int(RADIUS_M)}m_mean_bbox_h": safe_mean(bbox_h[r_idx]),
            f"radius_{int(RADIUS_M)}m_mean_aspect_ratio": safe_mean(aspect[r_idx]),

            f"radius_{int(RADIUS_M)}m_same_ura_zone_ratio": same_zone_ratio,
            f"radius_{int(RADIUS_M)}m_nearest_neighbor_dist": safe_min(r_dist_m),
            f"radius_{int(RADIUS_M)}m_mean_neighbor_dist": safe_mean(r_dist_m, default=-1.0),

            f"knn_{K_NEIGHBOURS}_mean_area": safe_mean(areas[k_idx]),
            f"knn_{K_NEIGHBOURS}_std_area": safe_std(areas[k_idx]),
            f"knn_{K_NEIGHBOURS}_mean_perimeter": safe_mean(perimeters[k_idx]),
            f"knn_{K_NEIGHBOURS}_mean_aspect_ratio": safe_mean(aspect[k_idx]),
            f"knn_{K_NEIGHBOURS}_mean_dist": safe_mean(k_dist_m, default=-1.0),
            f"knn_{K_NEIGHBOURS}_same_ura_zone_ratio": (
                float(np.mean(ura_zones[k_idx] == current_zone))
                if len(k_idx)
                else 0.0
            ),
        }

        for zone in unique_zones:
            rec[f"radius_{int(RADIUS_M)}m_ura_{zone}_frac"] = (
                float(np.mean(neighbour_zones == zone))
                if len(neighbour_zones)
                else 0.0
            )

            rec[f"nearest_ura_{zone}_dist_m"] = float(nearest_zone_dist[zone][i])

        rec[f"radius_{int(RADIUS_M)}m_dominant_ura_zone"] = dominant_zone

        records.append(rec)

    neigh = pd.DataFrame(records)

    dominant_dummies = pd.get_dummies(
        neigh[f"radius_{int(RADIUS_M)}m_dominant_ura_zone"],
        prefix=f"radius_{int(RADIUS_M)}m_dominant_ura"
    )

    neigh = pd.concat(
        [
            neigh.drop(columns=[f"radius_{int(RADIUS_M)}m_dominant_ura_zone"]),
            dominant_dummies
        ],
        axis=1
    )

    enhanced = features.merge(neigh, on="building_id", how="left")

    neigh = neigh.fillna(0.0)
    enhanced = enhanced.fillna(0.0)

    neigh.to_csv("neighbourhood_features.csv", index=False)
    enhanced.to_csv("features_with_neighbourhood.csv", index=False)

    print("\nSaved neighbourhood_features.csv")
    print("Saved features_with_neighbourhood.csv")
    print(f"Created {len(neigh.columns) - 1} neighbourhood feature columns")

    print("\nColumns containing 'zone':")
    for c in enhanced.columns:
        if "zone" in c.lower():
            print(c)


if __name__ == "__main__":
    main()