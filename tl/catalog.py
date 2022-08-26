#!/usr/bin/env python


"""
This modules handles the catalog
"""


# Python Standard Library

# Other dependencies
import numpy as np
import pandas as pd

# Local files


__author__ = 'Leonardo van der Laat'
__email__  = 'laat@umich.edu'


def read(filepath):
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df.time)
    return df


def filter(
    df,
    magnitude_min=None,
    magnitude_max=None,
    depth_min=None,
    depth_max=None,
    n_events=None
):
    if magnitude_min is not None:
        df = df[df.magnitude >= magnitude_min]
    if magnitude_max is not None:
        df = df[df.magnitude <= magnitude_max]
    if depth_min is not None:
        df = df[df.depth >= depth_min]
    if depth_max is not None:
        df = df[df.depth <= depth_max]

    if n_events is not None:
        if len(df) < n_events:
            n_events = len(df)
        df.sort_values(by='magnitude', inplace=True, ignore_index=True)
        index = np.linspace(0, len(df)-1, n_events, dtype=int)
        df = df.loc[index]
    return df


if __name__ == '__main__':
    main()
