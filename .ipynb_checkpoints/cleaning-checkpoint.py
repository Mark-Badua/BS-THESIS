import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def dropped_pts(data: pd.DataFrame, col: str, start_date = '2019-12-31 12', end_date ='2020-01-01 06'):
    df = data[col].dropna()
    df_ny = df[start_date:end_date]
    df_ny_roc = (df_ny.to_numpy()[1:] - df_ny.to_numpy()[:-1])/((df_ny.index[1:] - df_ny.index[:-1])/np.timedelta64(1,'m'))
    df_ny_roc_max = df_ny_roc.max()
    df_ny_roc_min = df_ny_roc.min()
    data = []
    timestamps = []
    for pt1, pt2, pt3, d1, d2, d3 in zip(df[:-2].to_numpy(), df[1:-1].to_numpy(),
                                     df[2:].to_numpy(), df.index[:-2],
                                     df.index[1:-1], df.index[2:]):
        if (((d2-d1)/np.timedelta64(1,'m')) > 10) or (((d3-d2)/np.timedelta64(1,'m')) > 10): 
            continue
        else:
            m1 = (pt2-pt1)/((d2-d1)/np.timedelta64(1,'m'))
            m2 = (pt3-pt2)/((d3-d2)/np.timedelta64(1,'m'))
            if (m1*m2 <0) and (m1 >=0):
                if ((m1 > (df_ny_roc_max)*5) or (m2 < (df_ny_roc_min*5))):
                    data.append(pt2)
                    timestamps.append(d2)
    return data, timestamps

def remove_rows(ts: pd.Timestamp, omitted_timestamps: list[pd.Timestamp]):
    for t in omitted_timestamps:
        if (t == ts):
            return False
    return True

def clean(df: pd.DataFrame, col:str):
    x, t = dropped_pts(df, col)
    truth_val = df.dropna().index.to_series().apply(lambda ts: remove_rows(ts, t))
    cleaned_df = df.dropna()[truth_val]
    removed_val = df.dropna()[~truth_val]
    return cleaned_df[col], removed_val[col]