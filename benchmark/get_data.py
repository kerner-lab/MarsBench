import pandas as pd
import wandb

api = wandb.Api()


def expand_dict_columns(df):
    summary_df = pd.json_normalize(df["summary"])
    config_df = pd.json_normalize(df["config"])

    df_expanded = pd.concat([df.drop(["summary", "config"], axis=1), summary_df, config_df], axis=1)
    return df_expanded


# [MarsBenchClassificationWithSeeds, MarsBenchSegmentationWithSeeds]
def get_data(run_name, columns=None):
    runs = api.runs("irish-mehta-arizona-state-university/" + run_name)

    summary_list, config_list, name_list = [], [], []

    for run in runs:

        summary_list.append(run.summary._json_dict)

        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

        name_list.append(run.name)

    runs_df = pd.DataFrame({"summary": summary_list, "config": config_list, "name": name_list})
    runs_df = expand_dict_columns(runs_df)
    if columns:
        runs_df = runs_df[columns]
    return runs_df
