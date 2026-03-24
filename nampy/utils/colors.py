import matplotlib as mpl
import numpy as np


def color_fader(c_1, c_2, mix=0):
    """
    Mix two colors as defined by mix in [0, 1].
    """
    c_1 = np.array(mpl.colors.to_rgb(c_1))
    c_2 = np.array(mpl.colors.to_rgb(c_2))
    if isinstance(mix, np.ndarray):
        cols = []
        for i in range(len(mix)):
            cols.append(mpl.colors.to_hex((1 - mix[i]) * c_1 + mix[i] * c_2))
        return cols
    return mpl.colors.to_hex((1 - mix) * c_1 + mix * c_2)


def color_bounds(values):
    """
    Helper for plotting MRF smooths.
    """
    interval = np.linspace(0, 1, 100)
    min_v = min(values)[0]
    max_v = max(values)[0]
    mapped = min_v + ((max_v - min_v) / 1 - 0) * interval
    return mapped
