#!/usr/bin/env python


"""
"""


# Python Standard Library

# Other dependencies
import numpy as np
import pandas as pd

# Local files


__author__ = 'Leonardo van der Laat'
__email__  = 'laat@umich.edu'


def get_scores(y_test, y_pred):
    misfits = np.sqrt(np.sum((y_pred - y_test)**2, axis=1))

    sum_errs = np.sum((y_test-y_pred)**2, axis=0)
    stdev = np.sqrt(1/(y_test.shape[0]-2) * sum_errs)

    interval = 1.96 * stdev
    return misfits, interval, stdev


def results_to_df(y_true, y_pred, misfits, evid_test):
    df_true = pd.DataFrame(y_true, columns='x y z'.split())
    df_true['misfit'] = misfits
    df_true['eventid'] = evid_test

    df_pred = pd.DataFrame(y_pred, columns='x y z'.split())
    df_pred['misfit'] = misfits
    df_pred['eventid'] = evid_test
    return df_true, df_pred


if __name__ == '__main__':
    main()
