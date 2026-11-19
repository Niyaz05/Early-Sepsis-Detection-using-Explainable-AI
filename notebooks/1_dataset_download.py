from pathlib import Path
import shutil
import kagglehub

# Project root: parent directory of this script's directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Destination directory
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Download dataset from Kaggle
download_path = Path(
    kagglehub.dataset_download(
        "tea340yashjoshi/sepsis-prediction-dataset"
    )
)

print("Downloaded dataset location:", download_path)

# Find CSV files in the downloaded directory
csv_files = list(download_path.rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {download_path}"
    )

# If there are multiple CSVs, inspect them before choosing one
if len(csv_files) > 1:
    print("Multiple CSV files found:")
    for file in csv_files:
        print(file)
    raise RuntimeError(
        "Multiple CSV files found. Select the correct file explicitly."
    )

# Copy the dataset to the desired location
destination = RAW_DIR / "dataset.csv"
shutil.copy2(csv_files[0], destination)

print("Dataset saved successfully:", destination)
print("File size (MB):", round(destination.stat().st_size / (1024**2), 2))
