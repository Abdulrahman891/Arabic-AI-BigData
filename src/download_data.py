from datasets import load_dataset
import pandas as pd
import os

dataset = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")

os.makedirs("data/raw", exist_ok=True)

for split_name in dataset.keys():
    df = dataset[split_name].to_pandas()
    output_path = f"data/raw/{split_name}.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_path}")
