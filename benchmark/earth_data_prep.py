import json
import os

import pandas as pd

columns = ["seed", "model", "dataset", "partition name", "test metric"]
datasets = ["mb-surface_cls", "mb-domars16k", "mb-atmospheric_dust_cls_edr", "mb-frost_cls"]
df = pd.DataFrame(columns=columns)

satmae_results = "/data/hkerner/MarsBench/results/eo_results"

seed = 42

for each_file in os.listdir(os.path.join(satmae_results)):
    current_df = pd.read_csv(os.path.join(satmae_results, each_file))

    for index, row in current_df.iterrows():
        current_partition = each_file.split("_")[-1][:4]
        if "CROMA" in each_file:
            current_model = "CROMA_ViT-L/16"
        else:
            current_model = "SatMAE_ViT-L/16"
        current_dataset = row["dataset"]
        current_f1score = float(row["F1-Score"])
        current_row = [seed, current_model, current_dataset, current_partition, current_f1score]
        df.loc[len(df)] = current_row

satmae_results = "/data/hkerner/MarsBench/results/bimal_croma/completed"

seed = 42

for each_file in os.listdir(os.path.join(satmae_results)):
    current_df = pd.read_csv(os.path.join(satmae_results, each_file))

    for index, row in current_df.iterrows():
        current_partition = each_file.split("_")[-1][:4]
        current_model = "prithvi_eo_v1_100"
        current_dataset = row["dataset"]
        current_f1score = float(row["F1-Score"])
        current_row = [seed, current_model, current_dataset, current_partition, current_f1score]
        df.loc[len(df)] = current_row

# Mapping Data and Model
with open("mappings.json", "r") as f:
    mappings = json.load(f)
df["dataset"] = df["dataset"].map(mappings["data"])

# Add ViT-L/16
vitl = pd.read_csv("data/classification_partition_filtered.csv")
vitl = vitl[vitl["model"] == "ViT-L/16"][df.columns]
df = pd.concat([df, vitl], ignore_index=True)

df = df[df["dataset"].isin(datasets)]

df.to_csv("data/earth_data.csv", index=False)
