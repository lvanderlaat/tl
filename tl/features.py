#!/usr/bin/env python


"""
"""


# Python Standard Library
import gc
import itertools
import logging

# Other dependencies
import numpy as np
import pandas as pd
import scipy.signal
try:
    import ray.util.multiprocessing as multiprocessing
except:
    import multiprocessing

from scipy.special import lambertw

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


def _engineer_feature(df, transformations, pair_keys):
    def R(ratio):
        return ratio

    def W(ratio):
        return lambertw(ratio).real

    functions = dict(
        R=R,
        sqrt=np.sqrt,
        log=np.log,
        W=W,
    )

    key_i, key_j = pair_keys
    key = f'{key_i}_{key_j}'

    ratio = df[key_i].values / df[key_j].values

    keys, features = [], []
    for transformation in transformations:
        keys.append(f'{key}_{transformation}')
        features.append(functions[transformation](ratio))
    return keys, features


def engineer(
    df, meta, ratio_same_station, ratio_diff_bands, transformations,
    max_workers
):
    n_amplitudes = len(meta)
    pairs_keys = []
    station_i, station_j = [], []
    channel_i, channel_j = [], []
    freqmin_i, freqmax_i = [], []
    freqmin_j, freqmax_j = [], []
    for i in range(0, n_amplitudes-1):
        for j in range(i+1, n_amplitudes):
            amplitude_i = meta.iloc[i]
            amplitude_j = meta.iloc[j]

            # Do not compute ratios between the same station
            if not ratio_same_station:
                if amplitude_i.station == amplitude_j.station:
                        continue
            # Do not compute ratios between different bands
            if not ratio_diff_bands:
                if (amplitude_i.freqmin != amplitude_j.freqmin) or \
                   (amplitude_i.freqmax != amplitude_j.freqmax):
                    continue

            pairs_keys.append((amplitude_i.key, amplitude_j.key))

            station_i.append(amplitude_i.station)
            station_j.append(amplitude_j.station)

            channel_i.append(amplitude_i.channel)
            channel_j.append(amplitude_j.channel)

            freqmin_i.append(amplitude_i.freqmin)
            freqmin_j.append(amplitude_j.freqmin)

            freqmax_i.append(amplitude_i.freqmax)
            freqmax_j.append(amplitude_j.freqmax)

    # Process
    with multiprocessing.Pool(max_workers) as pool:
        results = pool.starmap(
            _engineer_feature,
            zip(
                itertools.repeat(df),
                itertools.repeat(transformations),
                pairs_keys
            )
        )

    # Remove amplitudes, no need to save on file
    df.drop(labels=meta.key.tolist(), axis=1, inplace=True)

    # Add features to dataframe
    data = dict()
    metadata = dict(
        key=[],
        station_i=[],
        station_j=[],
        channel_i=[],
        channel_j=[],
        freqmin_i=[],
        freqmin_j=[],
        freqmax_i=[],
        freqmax_j=[],
        transformation=[]
    )
    for i, pair_keys in enumerate(pairs_keys):
        keys, values = results[i][0], results[i][1]
        data.update(dict(zip(keys, values)))

        metadata['key'].extend(keys)
        metadata['station_i'].extend([station_i[i]]*len(keys))
        metadata['station_j'].extend([station_j[i]]*len(keys))
        metadata['channel_i'].extend([channel_i[i]]*len(keys))
        metadata['channel_j'].extend([channel_j[i]]*len(keys))
        metadata['freqmin_i'].extend([freqmin_i[i]]*len(keys))
        metadata['freqmin_j'].extend([freqmin_j[i]]*len(keys))
        metadata['freqmax_i'].extend([freqmax_i[i]]*len(keys))
        metadata['freqmax_j'].extend([freqmax_j[i]]*len(keys))
        metadata['transformation'].extend([s.split('_')[-1] for s in keys])

    metadata = pd.DataFrame(metadata)
    df = pd.concat([df, pd.DataFrame(data, index=df.index)], axis=1)
    return metadata, df


def filter(
    meta, stations, channels, bands,
    ratio_diff_bands, ratio_same_station, transformations
):
    # Get rid of undesired channels/stations
    meta = meta[(meta.station_i.isin(stations)) & (meta.station_j.isin(stations))]
    meta = meta[(meta.channel_i.isin(channels)) & (meta.channel_j.isin(channels))]

    # Remove ratios between differnt bands if so desired
    if not ratio_diff_bands:
        meta = meta.drop(
            meta[
                (meta.freqmin_i != meta.freqmin_j) |
                (meta.freqmax_i != meta.freqmax_j)
            ].index
        )
    # Remove ratios between the same station if so desired
    if not ratio_same_station:
        meta = meta.drop(meta[(meta.station_i == meta.station_j)].index)

    # Remove features not in transformations indicated
    meta = meta.drop(meta[~meta.transformation.isin(transformations)].index)

    # Remove features which bands are not in config file
    meta['band_i'] = list(zip(meta.freqmin_i, meta.freqmax_i))
    meta['band_j'] = list(zip(meta.freqmin_j, meta.freqmax_j))
    bands = [tuple(band) for band in bands]
    meta = meta.drop(meta[~meta.band_i.isin(bands)].index)
    meta = meta.drop(meta[~meta.band_j.isin(bands)].index)
    return meta


if __name__ == '__main__':
    pass
