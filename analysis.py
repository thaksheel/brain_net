import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

files = [
    "./exports/rslt_hcpgender_exp_ng.xlsx",
    "./exports/rslt_hcpage_exp_ng.xlsx",
    "./exports/rslt_hcptask_exp_ng.xlsx",
]
files = [
    "./exports/exports_msi/rslt_mutag_exp_stnd.xlsx",
    "./exports/exports_msi/rslt_nci1_exp_stnd.xlsx",
    "./exports/exports_msi/rslt_nci109_exp_stnd.xlsx",
    "./exports/exports_msi/rslt_proteins_exp_stnd.xlsx",
]
files = [
    "./exports/exports_msi/tu_results_mutag.xlsx",
    "./exports/exports_msi/tu_results_proteins.xlsx",
    "./exports/exports_msi/tu_results_nci1.xlsx",
    "./exports/exports_msi/tu_results_nci109.xlsx",
]
files = [
    "./exports/rslt_a116_stnd.xlsx",
    "./exports/rslt_p264_stnd.xlsx",
    "./exports/rslt_s100_stnd.xlsx",
]
files = [
    "./exports/rslt_a116_th.xlsx",
    "./exports/rslt_p264_th.xlsx",
    "./exports/rslt_s100_th.xlsx",
]
# dataset_names = [c.split("rslt_")[1].split("_exp")[0] for c in files]
dataset_names = [c.split("rslt_")[1].split("_th")[0] for c in files]
# dataset_names = [c.split("tu_results_")[1].strip(".xlsx") for c in files]
dfs = [pd.read_excel(fie) for fie in files]
summary = [
    {
        "acc": df.accuracy_test.max(),
        "f1": df.f1_macro_test.max(),
    }
    for df in dfs
]
summaries = []
for j, f in enumerate(files):
    df = pd.read_excel(f)
    seeds = df.seed.unique()
    series = []
    for seed in seeds:
        df_s = df[df.seed == seed]
        idx = df_s.accuracy_val.idxmax()
        df_ss = df_s.loc[idx]
        series.append(df_ss)
    df_best = pd.DataFrame(series)
    fields = ["accuracy", "loss", "f1_macro", "sensitivity", "specificity", "pr_auc"]
    cols = []
    for col in df.columns:
        for f in fields:
            if f in col:
                cols.append(col)
    df_best: pd.DataFrame = df_best[cols]
    summaries.append(
        pd.DataFrame(
            {
                "mean": df_best.mean(0),
                "std": df_best.std(0),
                "count": df_best.count(0),
                "dataset": [dataset_names[j] for _ in df_best.count(0)],
            }
        )
    )
df_summary = pd.concat(summaries)
df_long = df_summary.reset_index().rename(columns={"index": "metric"})
df_wide = df_long.pivot(index="metric", columns="dataset", values=["mean", "std"])
df_wide.columns = [f"{ds}_{stat}" for stat, ds in df_wide.columns]
df_wide.to_excel("./exports/analysis/ng_aggr.xlsx")

print(df_best.head())
print("END")
