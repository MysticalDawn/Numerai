# Numerai Data Analysis V5.2

This repository contains code for exploratory data analysis and model building for the Numerai tournament using the v5.2 dataset.

## Setup

To get started, you'll need to install the required dependencies:

```bash
pip install numerapi pandas jupyter
```

## Usage

The main analysis is contained in `main.ipynb`. You can run it using Jupyter Notebook or JupyterLab:

```bash
jupyter notebook main.ipynb
```

## Project Structure

- `data.py`: Helper script to download and load the Numerai v5.2 dataset.
- `main.ipynb`: Jupyter notebook for data exploration and analysis.
- `model.py`: (Placeholder) Module for model definitions.
- `preprocessing.py`: (Placeholder) Module for data preprocessing functions.
- `v5.2/`: Directory containing the downloaded dataset (ignored by git).
- `data/`: Directory for storing processed data (ignored by git).

## Getting Data

The `data.py` script uses `numerapi` to download the `v5.2/train.parquet` file if it's not already present.
