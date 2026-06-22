import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer

print("=" * 80)
print("URA ZONING MATCHING: Buildings to Official Zones")
print("=" * 80)

URA_FILE = "ura_master_plan_2019_land_use.geojson"
COORDINATES_FILE = "coordinates.csv"
OUTPUT_FILE = "building_zones_ura_FIXED.csv"

print("\n[1/4] Loading URA zoning polygons...")
gdf_zones = gpd.read_file(URA_FILE)

print(f"Loaded {len(gdf_zones)} zoning polygons")
print("Original CRS:", gdf_zones.crs)
print("Columns:", list(gdf_zones.columns))

if gdf_zones.crs is None:
    gdf_zones = gdf_zones.set_crs("EPSG:4326")

gdf_zones = gdf_zones.to_crs("EPSG:4326")

print("\n[2/4] Loading building SVY21 coordinates...")
df_buildings = pd.read_csv(COORDINATES_FILE)

print(f"Loaded {len(df_buildings):,} buildings")
print(df_buildings.head())

print("\n[3/4] Converting SVY21 x,y to WGS84 longitude,latitude...")

transformer = Transformer.from_crs(
    "EPSG:3414",
    "EPSG:4326",
    always_xy=True
)

lon, lat = transformer.transform(
    df_buildings["x"].values,
    df_buildings["y"].values
)

df_buildings["longitude"] = lon
df_buildings["latitude"] = lat

print(f"Latitude range: {df_buildings['latitude'].min():.4f} to {df_buildings['latitude'].max():.4f}")
print(f"Longitude range: {df_buildings['longitude'].min():.4f} to {df_buildings['longitude'].max():.4f}")

print("\n[4/4] Matching buildings to URA polygons using spatial join...")

gdf_buildings = gpd.GeoDataFrame(
    df_buildings.copy(),
    geometry=[
        Point(xy) for xy in zip(
            df_buildings["longitude"],
            df_buildings["latitude"]
        )
    ],
    crs="EPSG:4326"
)

joined = gpd.sjoin(
    gdf_buildings,
    gdf_zones,
    how="left",
    predicate="within"
)

zone_col = "LU_DESC"

joined["ura_zone"] = joined[zone_col].fillna("Unmatched")

df_result = joined[
    [
        "building_id",
        "x",
        "y",
        "z",
        "latitude",
        "longitude",
        "ura_zone"
    ]
].copy()

df_result.to_csv(OUTPUT_FILE, index=False)

matched = (df_result["ura_zone"] != "Unmatched").sum()
unmatched = (df_result["ura_zone"] == "Unmatched").sum()
total = len(df_result)

print("\n" + "=" * 80)
print("URA ZONING RESULTS")
print("=" * 80)

print(f"Total buildings: {total:,}")
print(f"Matched: {matched:,} ({matched / total * 100:.2f}%)")
print(f"Unmatched: {unmatched:,} ({unmatched / total * 100:.2f}%)")

print("\nZone Types Found:")
print("-" * 80)

zone_counts = df_result["ura_zone"].value_counts()

for zone, count in zone_counts.items():
    percentage = count / total * 100
    print(f"{zone:40} {count:6,} buildings ({percentage:5.2f}%)")

print(f"\nSaved fixed results to: {OUTPUT_FILE}")
print("\nDone.")