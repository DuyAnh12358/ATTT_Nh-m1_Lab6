import pandas as pd
import numpy as np
import glob
import os

def load_and_merge_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "*.csv")

    files = glob.glob(data_path)
    print("FILES FOUND:", files)

    df_list = []

    for file in files[:2]:
        print("Loading:", file)

        df = pd.read_csv(
            file,
            nrows=100000,
            low_memory=False
        )

        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)
    return df

def clean_data(df):
    df.columns = df.columns.str.strip()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df = df.fillna(df.median(numeric_only=True))

    df = df.astype('float32', errors='ignore')

    return df

def remove_useless(df):
    df = df.drop_duplicates()

    nunique = df.nunique()
    zero_cols = nunique[nunique <= 1].index
    df.drop(columns=zero_cols, inplace=True)

    return df

def optimize_memory(df):
    for col in df.select_dtypes(include=['float64']):
        df[col] = df[col].astype('float32')

    for col in df.select_dtypes(include=['int64']):
        df[col] = pd.to_numeric(df[col], downcast='integer')

    return df

def preprocess_pipeline():
    df = load_and_merge_data()
    df = clean_data(df)
    df = remove_useless(df)
    df = optimize_memory(df)

    return df