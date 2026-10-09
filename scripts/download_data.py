
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

# Public MovieLens 1M zip file.
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

# __file__ is scripts/download_data.py.
# parents[1] moves from scripts/ back to repo root.
root = Path(__file__).resolve().parents[1]

# Store downloaded files under data/raw.
raw_dir = root / "data" / "raw"
zip_path = raw_dir / "ml-1m.zip"
target_dir = raw_dir / "ml-1m"

# Create data/raw if it does not exist.
raw_dir.mkdir(parents=True, exist_ok=True)

# Download only if the zip is missing.
if not zip_path.exists():
    print("Downloading:", URL)
    urlretrieve(URL, zip_path)

# Extract only if the extracted folder is missing.
if not target_dir.exists():
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

# Validate that the three expected raw files exist.
for name in ["ratings.dat", "movies.dat", "users.dat"]:
    path = target_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    print(name, path.stat().st_size, "bytes")

print("ready:", target_dir)
