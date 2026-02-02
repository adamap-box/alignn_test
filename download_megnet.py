"""Download MEGNet dataset and save to megnet_data folder."""
import os
import json
from jarvis.db.figshare import data as jdata

# Create output folder
output_dir = "megnet_data"
os.makedirs(output_dir, exist_ok=True)

# Download MEGNet dataset
print("Downloading MEGNet dataset...")
megnet = jdata("megnet")
print(f"Downloaded {len(megnet)} samples")

# Save as JSON
output_file = os.path.join(output_dir, "megnet.json")
print(f"Saving to {output_file}...")
with open(output_file, "w") as f:
    json.dump(megnet, f)

print(f"Done! Saved {len(megnet)} samples to {output_file}")

# Show sample entry
print("\nSample entry keys:", list(megnet[0].keys()))
print("Sample ID:", megnet[0].get("id", "N/A"))
print("Sample e_form:", megnet[0].get("e_form", "N/A"))
