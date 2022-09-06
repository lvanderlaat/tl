#!/usr/bin/env python


""" Synth
Generates synthetic data based on the amplitude decay model
"""


# Python Standard Library

# Other dependencies
import pandas as pd
import numpy as np

# Local files
from .projection import geographic_to_cartesian
from .features import engineer
from .plot import synth_misfit


__author__ = 'Leonardo van der Laat'
__email__  = 'laat@umich.edu'


def amplitudes(df, meta, f, Q, beta, alpha, source_amplitude=True):
    Ao = 1
    if source_amplitude:
        Ao = np.exp(df.magnitude)

    B  = (np.pi*f)/(Q*beta)

    # Synthetic amplitudes
    for i, row in meta.iterrows():
        distance = np.sqrt(
            (df.x - row.x)**2 + (df.y - row.y)**2 + (df.z - row.z)**2
        )
        df[row.key] = Ao*np.exp(-B*distance)/distance**alpha
    return df


def distribution(
    meta,
    model, scaler_X, scaler_y,
    epsg,
    lonmin, lonmax, latmin, latmax, zmin, zmax, step, z_plot, y_plot,
    f, Q, beta, alpha,
    transformations,
    max_workers
):
    # Predict grid
    xmin, ymin = geographic_to_cartesian(lonmin, latmin, epsg)
    xmax, ymax = geographic_to_cartesian(lonmax, latmax, epsg)

    # Create grids
    # Map view
    xh = np.arange(xmin, xmax + step, step)
    yh = np.arange(ymin, ymax + step, step)

    xx, yy = np.meshgrid(xh, yh)

    z = np.ones(xx.flatten().shape)*z_plot

    dfh = pd.DataFrame(dict(x=xx.flatten(), y=yy.flatten(), z=z))

    # Profile view
    zv = np.arange(zmax, zmin-step, -step)

    xx, zz = np.meshgrid(xh, zv)
    # y_plot = (meta.y.min() + meta.y.max())/2
    y = np.ones(xx.flatten().shape)*y_plot

    dfv = pd.DataFrame(dict(x=xx.flatten(), y=y, z=zz.flatten()))

    for df in [dfh, dfv]:
        df = amplitudes(df, meta, f, Q, beta, alpha, source_amplitude=False)

        metadata, features = engineer(
            df, meta, False, False, transformations, max_workers
        )

        X = features[metadata.key].values
        y = features[['x', 'y', 'z']].values

        # Scale features
        X = scaler_X.transform(X)

        # Predict
        y_pred = model.predict(X)

        y_pred = scaler_y.inverse_transform(y_pred)
        # Error
        df['misfit'] = np.sqrt(np.sum((y_pred - y)**2, axis=1))
    # return dfh, dfv, xh, yh, zv, y_plot
    return dfh, dfv, xh, yh, zv


if __name__ == '__main__':
    pass
