"""Create ENVSDD_test.json covering ALL datasets from test_metadata.csv."""
import pandas as pd, json, os
from collections import Counter

meta = pd.read_csv("data/test/test_set/test_metadata.csv")
print(f"Metadata: {len(meta)} samples")

gen_map = {
    ("ata", "audioldm1"): "fake_ata_01",
    ("ata", "audioldm2"): "fake_ata_02",
    ("tta", "audiogen"): "fake_tta_01",
    ("tta", "audioldm1"): "fake_tta_02",
    ("tta", "audioldm2"): "fake_tta_03",
    ("tta", "audiolcm"): "fake_unknown_01",
    ("tta", "tangoflux"): "fake_unknown_02",
}

entries = []
entries_5c = []
missing = 0

for _, row in meta.iterrows():
    audio_path = f"data/test/test_set/audio/{row['wavename']}"
    if not os.path.exists(audio_path):
        missing += 1
        continue

    # Binary
    label_bin = "real" if row["faketype"] == "real" else "fake"
    entries.append({"audio": audio_path, "label": label_bin})

    # 5-class
    if row["faketype"] == "real":
        label_5c = "real"
    else:
        label_5c = gen_map.get((row["faketype"], row["generator"]), "fake_unknown")
    entries_5c.append({"audio": audio_path, "label": label_5c})

print(f"Missing files: {missing}")

# Save binary
with open("data/label/beats/ENVSDD_test.json", "w") as f:
    json.dump(entries, f, indent=2)
counts = Counter(e["label"] for e in entries)
print(f"\nENVSDD_test.json: {len(entries)} samples")
for k, v in sorted(counts.items()):
    print(f"  {k}: {v}")

# Save 5-class
with open("data/label/beats/ENVSDD_test_5class.json", "w") as f:
    json.dump(entries_5c, f, indent=2)
counts5 = Counter(e["label"] for e in entries_5c)
print(f"\nENVSDD_test_5class.json: {len(entries_5c)} samples")
for k, v in sorted(counts5.items()):
    print(f"  {k}: {v}")

# Per source dataset
print("\nPer source dataset:")
for ds in sorted(meta["source dataset"].unique()):
    sub = meta[meta["source dataset"] == ds]
    n_real = (sub["faketype"] == "real").sum()
    n_fake = (sub["faketype"] != "real").sum()
    print(f"  {ds:20s}  total={len(sub):6d}  real={n_real:5d}  fake={n_fake:5d}")
