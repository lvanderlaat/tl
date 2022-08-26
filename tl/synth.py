#!/usr/bin/env python


""" Synth
Generates synthetic data based on the amplitude decay model
"""


# Python Standard Library

# Other dependencies
import numpy as np

# Local files


__author__ = 'Leonardo van der Laat'
__email__  = 'laat@umich.edu'


def get_data(df, meta, f, Q, beta, alpha, source_amplitude=False):
    Ao = 1
    if source_amplitude:
        Ao = np.exp(df.magnitude)

    B  = (np.pi*f)/(Q*beta)

    # Synthetic amplitudes
    for i, row in meta.iterrows():
        station = row.station
        distance = np.sqrt(
            (df.x - row.x)**2 + (df.y - row.y)**2 + (df.z - row.z)**2
        )
        df[station] = Ao*np.exp(-B*distance)/distance**alpha

    # Ratios
    n_features = len(meta)
    feature_keys = []
    for i in range(0, n_features-1):
        for j in range(i+1, n_features):
            feature_i = meta.iloc[i]
            feature_j = meta.iloc[j]
            pair = f'{feature_i.station}_{feature_j.station}'
            df[pair] = df[feature_i.station] / df[feature_j.station]
            df[pair+'_sqrt'] = np.sqrt(df[pair])
            df[pair+'_log']  = np.log(df[pair])
            feature_keys.extend([pair, pair+'_sqrt', pair+'_log'])

    # To NumPy arrays for SciKit-Learn
    X = df[feature_keys].values
    y = df[['x', 'y', 'z']].values
    return X, y


def get_misfit(df, meta, f, Q, beta, alpha, model, scaler):
    X, y_true = get_data(df, meta, f, Q, beta, alpha)

    X = scaler.transform(X)

    # Predict
    y_pred = model.predict(X)

    # Error
    df['misfit'] = np.sqrt(np.sum((y_pred - y_true)**2, axis=1))
    return


if __name__ == '__main__':
    pass
