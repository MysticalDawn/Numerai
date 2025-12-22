from numerapi import NumerAPI
import pandas as pd


VERSION = "v5.2"


def get_data(version: str = VERSION):
    napi = NumerAPI()
    napi.download_dataset(f"{version}/train.parquet")
    return pd.read_parquet(f"{version}/train.parquet")
