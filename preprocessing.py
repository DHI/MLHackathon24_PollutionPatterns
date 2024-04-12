import pandas as pd
import numpy as np

from sklearn.preprocessing import FunctionTransformer

# Including harmonic signals
def sin_encoder(period):
        return FunctionTransformer(lambda x: np.sin((2 * np.pi* x)/period))

def cos_encoder(period):

    return FunctionTransformer(lambda x: np.cos((2 * np.pi* x)/period))

def adding_remaining_features(df):

    df.index = pd.to_datetime(df.index, utc=True)

    pollutants = ["TOC", "TN", "TP", "SS"]

    df["sine"] = sin_encoder(24).fit_transform(df.index.hour)
    df["cosine"] = cos_encoder(24).fit_transform(df.index.hour)

    df["ones"] = 1

    # Including day of week
    df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)

    # Taking the logarithm of Pollutants
    logcn = lambda x: f"log{x}"
    logpollutants = []
    for p in pollutants:
        cn = logcn(p)
        df[cn] = np.log(1 + df[p].divide(df[p].max()))
        logpollutants.append(cn)

    return df

#The function
def dataProcessing_Chunks(inputdf,features,hours_ahead,hours_behind, offset=1):
    for feature in features:
        for i in range(offset,hours_ahead+1,1):
            inputdf[f'{feature}+{i}'] = inputdf[f'{feature}'].shift(-i)
    for feature in features:
        for i in range(offset,hours_behind+1,1):
            inputdf[f'{feature}-{i}'] = inputdf[f'{feature}'].shift(i)   
    inputdf.dropna(inplace=True)
    return inputdf
  