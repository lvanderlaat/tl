#!/usr/bin/env python


"""
"""


# Python Standard Library
import gc
import logging

# Other dependencies
import numpy as np
import pandas as pd
import scipy.signal

# Local files
from .obspy2numpy import st2windowed_data
from .pre_process import pre_process


__author__ = 'Leonardo van der Laat'
__email__  = 'laat@umich.edu'


def to_dataframe(channels, bands):
    channels.sort_values(by='station channel'.split())
    data = []
    for i, channel in channels.iterrows():
        for band in bands:
            datum = dict(
                station=channel.station,
                channel=channel.channel,
                latitude=channel.latitude,
                longitude=channel.longitude,
                x=channel.x,
                y=channel.y,
                z=channel.z,
                freqmin=band[0],
                freqmax=band[1],
                key=f'{channel.station}_{channel.channel}_{band[0]}_{band[1]}'
            )
            data.append(datum)
    return pd.DataFrame(data)


def extract_time_domain(
    st, inventory, freqmin, freqmax, stachas, filter_polynomials, method,
    window_length=None, overlap=0
):
    # Remove traces not requested
    st_stachas = []
    for tr in st:
        if (tr.stats.station, tr.stats.channel) not in stachas:
            st.remove(tr)
        else:
            st_stachas.append((tr.stats.station, tr.stats.channel))

    # Check if missing channels
    for stacha in stachas:
        if stacha not in st_stachas:
            logging.warning(f'{stacha} not in stream, {st[0].stats.starttime}.')

    # Pre-process
    pre_process(st, inventory, freqmin, freqmax)

    # Use the whole window if not specified otherwise
    if window_length is None:
        starttime = max(tr.stats.starttime for tr in st)
        endtime   = min(tr.stats.endtime   for tr in st)
        st = st.trim(starttime, endtime)
        window_length = (endtime - starttime)

    # Create windows
    _, _data_windowed = st2windowed_data(st, window_length, overlap)
    if _data_windowed is None: return None, None
    n_channels, n_windows, npts = _data_windowed.shape
    del _data_windowed; gc.collect()

    # Create empty array
    _X = np.empty((n_channels, n_windows, len(filter_polynomials)))
    for i, (b, a) in enumerate(filter_polynomials):
        _st = st.copy()

        # Filter the data
        for tr in _st:
            tr.data = scipy.signal.lfilter(b, a, tr.data)

        # Create windows
        utcdatetimes, data = st2windowed_data(_st, window_length, overlap)
        del _st; gc.collect()

        if data is None: return None, None

        datetimes = [u.datetime for u in utcdatetimes]

        # Measure amplitudes
        if method == 'peak-to-peak':
            x = np.max(data, axis=2) + np.abs(np.min(data, axis=2))
        elif method == 'rms':
            x = np.sqrt((data**2).mean(axis=2))

        del data; gc.collect()

        _X[:, :, i] = x
        del x; gc.collect()

    # Create matrix that matches the expected channels
    X = np.empty((len(stachas), n_windows, len(filter_polynomials)))
    X[:] = np.nan
    for tr, _x in zip(st, _X):
        idx = stachas.index((tr.stats.station, tr.stats.channel))
        X[idx] = _x
    del _X; gc.collect()

    X = np.hstack(X)
    return datetimes, X


if __name__ == '__main__':
    pass
