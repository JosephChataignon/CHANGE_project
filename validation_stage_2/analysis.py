# -*- coding: utf-8 -*-
"""
Group and analyze the results of all benchmarking runs in this folder.

Each subfolder named result-<monthDD>-<model_shortname>-<data_scale> contains the
output of one run of benchmarking.py. We only read summary.json from each of them
and deduce the run metadata from the folder name:

        result-oct19-miniLM-nonFT
              ^^^^^ ^^^^^^ ^^^^^
              date  model  scale

  - element 3 (index 2) of the dash-split name -> model shortname (string)
  - element 4 (index 3) of the dash-split name -> data scale factor (string,
    kept as-is even when it is a number; "nonFT" means not fine-tuned)

Goal: compare the different embedding models. Outputs (written next to the script):
  - combined_summary.csv ...... one row per run with metadata + hit rates
  - hit_rates_vs_scale.png .... one image with three figures: top-1, top-5 and
                                top-10 hit rates vs data scale, one color per model

Usage:
    python analysis.py
"""

import os
import json
import glob

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ############################ Configuration ############################

# Correspondence between the model shortnames used in the folder names and the
# full model names we picked from. Matched case-insensitively.
# The order also defines the plot colors (one color per model).
SHORTNAME_TO_FULLNAME = {
    "minilm":   "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet":    "sentence-transformers/all-mpnet-base-v2",
    "qwen0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen8b":   "Qwen/Qwen3-Embedding-8B",
    "nemotron": "nvidia_llama-embed-nemotron-8b",
    "harrier":  "microsoft/harrier-oss-v1-0.6b",
}

# Order of the data scale on the x-axis of the plot. nonFT (not fine-tuned)
# counts as data scale 0 and stands first; the rest are increasing scales.
# Purely categorical: the values are used as labels with regular spacing.
SCALE_ORDER = ["nonFT", "0.0001", "0.1", "1", "10"]

# ############################ Loading the runs ############################

def parse_folder_name(folder_name):
    """Extract (date, model shortname, data scale factor) from a folder name.

    Folder names look like result-oct19-miniLM-nonFT; splitting on dashes gives
    4 elements and we care about the last two: the model shortname and the data
    scale factor (kept as a string even when it is a number).
    """
    parts = folder_name.split("-")
    if len(parts) != 4 or parts[0] != "result":
        return None  # not a standard result folder (e.g. tmp-result-*)
    date, shortname, scale = parts[1], parts[2], parts[3]
    return date, shortname, scale

def load_runs(base_dir):
    """Read summary.json from every standard result folder; return a DataFrame."""
    rows = []
    for summary_path in sorted(glob.glob(os.path.join(base_dir, "result-*", "summary.json"))):
        folder = os.path.basename(os.path.dirname(summary_path))
        parsed = parse_folder_name(folder)
        if parsed is None:
            print(f"Skipping (unexpected folder name): {folder}")
            continue
        date, shortname, scale = parsed
        full_name = SHORTNAME_TO_FULLNAME.get(shortname.lower())
        if full_name is None:
            print(f"Skipping (unknown shortname '{shortname}'): {folder}")
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        default = summary["hit_rates_default_threshold"]
        row = {
            "folder": folder,
            "date": date,
            "model_shortname": shortname,
            "model_fullname": full_name,
            "scale": scale,          # string on purpose, e.g. "0.1" or "nonFT"
            "timestamp": summary["timestamp"],
            "n_queries": summary["n_queries_evaluated"],
            "n_failed": summary["n_queries_failed"],
            "threshold": default["threshold"],
        }
        # hit rates at the default threshold...
        for k in (1, 5, 10):
            row[f"top_{k}_hit_rate"] = default[f"top_{k}"]
        # ...and at every candidate threshold (flat, for threshold comparison)
        for threshold, hit_rates in summary["hit_rates_by_threshold"].items():
            for k in (1, 5, 10):
                row[f"top_{k}@{float(threshold):.1f}"] = hit_rates[f"top_{k}"]
        rows.append(row)
    return pd.DataFrame(rows)

# ############################ Outputs ############################

def hit_rate_at(group, scale, column):
    """Hit rate at a given scale for one model (group); NaN if that scale was not run."""
    match = group.loc[group["scale"] == scale, column]
    return match.iloc[0] if len(match) else float("nan")

def make_scale_plot(df, plot_path):
    """One image, three figures: hit rate (top-1, top-5, top-10) vs data scale.

    The x-axis is categorical (nonFT, 0.0001, 0.1, 1, 10 with regular spacing),
    one color per model, so the models can be compared directly.
    """
    # Models in the configuration order; one color per model
    model_order = [name for name in SHORTNAME_TO_FULLNAME.values()
                   if name in df["model_fullname"].unique()]
    colors = {model: f"C{i}" for i, model in enumerate(model_order)}
    x = range(len(SCALE_ORDER))
    shortnames = {full: df[df["model_fullname"] == full]["model_shortname"].iloc[0]
                  for full in model_order}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, k in zip(axes, (1, 5, 10)):
        for model in model_order:
            group = df[df["model_fullname"] == model]
            # one y value per scale in SCALE_ORDER (NaN if that scale was not run)
            y = [hit_rate_at(group, scale, f"top_{k}_hit_rate")
                 for scale in SCALE_ORDER]
            ax.plot(x, y, marker="o", color=colors[model],
                    label=shortnames[model], linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(SCALE_ORDER)
        ax.set_xlabel("Data scale (nonFT = no fine-tuning)")
        ax.set_ylabel("Hit rate (threshold 0.5)")
        ax.set_title(f"Top-{k} hit rate")
        ax.grid(alpha=0.3)
    axes[0].legend(title="Model")
    fig.suptitle("FRAG search hit rates by embedding model and fine-tuning data scale")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

# ############################ Main ############################

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    df = load_runs(base_dir)
    if df.empty:
        raise SystemExit("No result-* folders with a summary.json found.")

    ## Flat table of all runs with their metadata and hit rates
    csv_path = os.path.join(base_dir, "combined_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Combined summary saved to: {csv_path} ({len(df)} runs)")

    ## Image with three figures: top-1, top-5 and top-10 hit rates vs data scale
    plot_path = os.path.join(base_dir, "hit_rates_vs_scale.png")
    make_scale_plot(df, plot_path)
    print(f"Plot saved to: {plot_path}")
