#!/usr/bin/env python

""" Inventory

This module handles with the station inventory

"""
# Python Standard Library
import datetime
import re

# Other dependencies
import matplotlib.pyplot as plt
from obspy import UTCDateTime
import pandas as pd


__author__ = 'Leonardo van der Laat'
__email__ = 'laat@umich.edu'


def to_dataframe(inventory, stations='.', channels='.'):
    stations_regex = ''
    for i, code in enumerate(stations):
        stations_regex += f'({code})'
        if i < len(stations) - 1:
            stations_regex += '|'

    channels_regex = ''
    for i, code in enumerate(channels):
        channels_regex += f'({code})'
        if i < len(channels) - 1:
            channels_regex += '|'

    data = []
    for network in inventory:
        for station in network:
            if not re.match(stations_regex, station.code):
                continue
            for channel in station.channels:
                if not re.match(channels_regex, channel.code):
                    continue
                data.append(dict(
                    network    = network.code,
                    station    = station.code,
                    channel    = channel.code,
                    longitude  = station.longitude,
                    latitude   = station.latitude,
                    z          = station.elevation,
                    # start_date = channel.start_date,
                    # end_date   = channel.end_date,
                ))
    df = pd.DataFrame(data)

    df.sort_values(by=['station', 'channel'], inplace=True)

    df.drop_duplicates(inplace=True)
    return df


def plot_timespan(inventory, stations='.', channels='.'):
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=.1,
        bottom=.1,
        right=.9,
        top=.9,
        wspace=.2,
        hspace=.2
    )

    ax = fig.add_subplot(111)

    count = 0
    yticklabels = []
    for network in inventory:
        for station in network:
            for channel in station.channels:
                end_date = channel.end_date
                if end_date is None:
                    end_date = UTCDateTime(2019, 1, 1)
                ax.plot(
                    [channel.start_date.datetime, end_date.datetime],
                    [count, count]
                )
                count += 1
                yticklabels.append(f'{station.code}.{channel.code}')
    ax.set_yticks(range(count))
    ax.set_yticklabels(yticklabels)
    plt.show()
    return


if __name__ == '__main__':
    pass
