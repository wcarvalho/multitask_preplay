#!/bin/bash
set -e

# Step 1: Download processed data from HuggingFace
python analysis/download_dataframes.py

# Step 2: Run analysis notebooks
jupyter nbconvert --to notebook --execute notebook.ipynb --inplace
