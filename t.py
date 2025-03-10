from numerapi import NumerAPI
import dotenv, os, json, pandas as pd, lightgbm as lgb

dotenv.load_dotenv('./.env')

P_ID = os.getenv('NUMERAI_PUBLIC_ID')
S_ID = os.getenv('NUMERAI_SECRET_ID')

# Initialize numerapi
napi = NumerAPI(public_id = P_ID, secret_key = S_ID)

all_datasets = napi.list_datasets()
#print(all_datasets)

# Set data version to one of the latest datasets
DATA_VERSION = "v5.0"

# download the feature metadata file
if not os.path.exists(f"{DATA_VERSION}/features.json"):
    napi.download_dataset(f"{DATA_VERSION}/features.json")

# read the metadata and display
feature_metadata = json.load(open(f"{DATA_VERSION}/features.json")) # all the feature sets and targets (the names)
#print(json.dumps(feature_metadata, indent=2))
for metadata in feature_metadata:
  print(metadata, len(feature_metadata[metadata]))

# display the feature sets
feature_sets = feature_metadata["feature_sets"]
#print(feature_sets["small"])
for feature_set in ["small", "medium", "all"]:
  print(feature_set, len(feature_sets[feature_set]))

# Only work with the medium feature set
medium_feature_set = feature_sets["medium"]

# Download the training data 
if not os.path.exists(f"{DATA_VERSION}/train.parquet"):  
  napi.download_dataset(f"{DATA_VERSION}/train.parquet")

# Load only the "medium" feature set 
train_feature_set = pd.read_parquet(
    f"{DATA_VERSION}/train.parquet",
    columns = ["era", "target"] + medium_feature_set
)
#print(train_feature_set.head())

# Downsample to every 4th era to reduce memory usage and speedup model training
train_feature_set_reduced = train_feature_set[train_feature_set["era"].isin(train_feature_set["era"].unique()[::4])]
#print(train_feature_set_reduced.head())

# https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
model = lgb.LGBMRegressor(
  n_estimators=2000,
  learning_rate=0.01,
  max_depth=5,
  num_leaves=2**5-1,
  colsample_bytree=0.1
)

model.fit(
  train_feature_set_reduced[medium_feature_set],
  train_feature_set_reduced["target"]
)