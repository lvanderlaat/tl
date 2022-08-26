""" Pre-process

Pre-process streams before feature extraction

"""


__author__ = 'Leonardo van der Laat'
__email__ = 'laat@umich.edu'


def pre_process(
    st,
    inventory,
    freqmin,
    freqmax,
    scale_factor=1e6,
    decimation_factor=2
):
    st.detrend()
    st.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
    st.remove_sensitivity(inventory)
    for tr in st:
        tr.data = tr.data * scale_factor

    if decimation_factor != 1:
        st.decimate(int(decimation_factor))
    return
