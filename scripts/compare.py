import json              # Import JSON so we can read saved metrics files.
from pathlib import Path # Import Path for portable filesystem paths.

root = Path(__file__).resolve().parents[1] # scripts/day06_compare.py → repo root.
artifact = root / "artifacts"              # Metrics are saved under artifacts/.
modes = ["zero", "status", "token", "vime"] # Fixed list of experiment modes to compare.

columns = [                         # Choose which metrics to print and in what order.
    "mode",                         # Name of experiment mode.
    "val_objective",                # Validation reconstruction / total objective.
    "val_mask_accuracy",            # VIME mask prediction accuracy; None for non-VIME modes.
    "next_genre_probe_micro_ap",    # Linear probe transfer metric.
    "embedding_mean_dim_std",       # Average per-dimension standard deviation of embeddings.
    "effective_rank",               # Representation effective rank diagnostic.
    "steps_per_second",             # Training speed metric.
]

rows = []                           # Store one metrics dictionary per mode.
for mode in modes:                  # Loop through zero/status/token/vime.
    path = artifact / f"day06_{mode}_metrics.json" # Build path to the metrics file.
    with open(path) as f:           # Open metrics JSON file.
        rows.append(json.load(f))   # Parse JSON into Python dict and append to rows.

print(" | ".join(columns))          # Print table header.
print("-" * 120)                    # Print visual separator.
for row in rows:                    # Print one row per experiment mode.
    values = []                     # Collect formatted cell values.
    for key in columns:             # Loop through selected metrics.
        value = row.get(key)        # Read metric value, or None if missing.
        if isinstance(value, float): # If value is float, format compactly.
            values.append(f"{value:.4f}") # Keep four decimal places.
        else:                       # Strings and None are printed directly.
            values.append(str(value)) # Convert to string for table output.
    print(" | ".join(values))       # Print row as pipe-separated text.