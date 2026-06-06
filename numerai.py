from numerapi import NumerAPI
import pandas as pd
import json
from ydata_profiling import ProfileReport
import seaborn as sns
import lightgbm as lgb
import sklearn
import numerblox
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from numerblox.meta import CrossValEstimator, make_meta_pipeline
from numerblox.ensemble import NumeraiEnsemble, PredictionReducer
from numerblox.download import NumeraiClassicDownloader
from numerblox.numerframe import create_numerframe
from numerblox.neutralizers import FeatureNeutralizer
import torch
import torch.nn as nn

DATASET_VERSION = "v5.2"


def get_dataset_version(napi):
    all_datasets = napi.list_datasets()
    dataset_versions = list(set(d.split("/")[0] for d in all_datasets))
    dataset_versions = sorted(dataset_versions, reverse=True)
    print("Available versions:\n", dataset_versions)
    return dataset_versions


def extract_dataset_files(all_datasets):
    current_version_files = [f for f in all_datasets if f.startswith(DATASET_VERSION)]
    print("Available", DATASET_VERSION, "files:\n", current_version_files)


def donwload_features(napi, donwload_flag=True, feature_metadata=None):
    if donwload_flag:
        napi.download_dataset(f"{DATASET_VERSION}/features.json")
    if feature_metadata is None:
        feature_metadata = json.load(open(f"./{DATASET_VERSION}/features.json"))
    for meta in feature_metadata:
        print(meta, len(feature_metadata[meta]))
    return feature_metadata


def select_dataset_size(feature_metadata, size="small"):
    feature_sets = feature_metadata["feature_sets"]
    return feature_sets[size]


def download_dataset(napi, feature_columns, download_flag=False):
    if download_flag:
        napi.download_dataset(f"{DATASET_VERSION}/train.parquet")
    data = pd.read_parquet(
        path=f"./{DATASET_VERSION}/train.parquet",
        columns=["era", "target"] + feature_columns,
    )
    return data


def main():
    global DATASET_VERSION
    napi = NumerAPI()
    dataset_versions = get_dataset_version(napi)
    DATASET_VERSION = dataset_versions[0]
    all_datasets = napi.list_datasets()
    extract_dataset_files(all_datasets)
    feature_metadata = donwload_features(napi, donwload_flag=False)
    print("Feature sets:", list(feature_metadata["feature_sets"].keys()))
    feature_columns = select_dataset_size(feature_metadata, size="small")
    data = download_dataset(napi, feature_columns)
    print(data.head())


if __name__ == "__main__":
    main()
