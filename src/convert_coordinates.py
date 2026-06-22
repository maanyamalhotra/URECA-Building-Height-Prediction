import pandas as pd
from pyproj import Transformer

df = pd.read_csv("coordinates.csv")
transformer = Transformer.from_epsg(3414, 4326)

print("Converting SVY21 → WGS84...")
lats, lons = transformer.transform(df['x'].values, df['y'].values)

df['latitude'] = lats
df['longitude'] = lons

print(df[['building_id', 'latitude', 'longitude']].head(10))
print(f"\nLat: {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
print(f"Lon: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")

df[['building_id', 'latitude', 'longitude']].to_csv("coordinates_latlon_CORRECT.csv", index=False)
print("\n Saved to coordinates_latlon_CORRECT.csv")
