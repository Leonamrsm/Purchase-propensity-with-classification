import pandas as pd
import numpy  as np

def get_first_order_statistics(df):
    # Central Tendency Metrics
    mean = pd.DataFrame(df.apply(np.mean)).T
    median = pd.DataFrame(df.apply(np.median)).T

    # Dispersion Metrics
    min_ = pd.DataFrame(df.apply(min)).T
    max_ = pd.DataFrame(df.apply(max)).T
    range_ = pd.DataFrame(df.apply(lambda x: x.max() - x.min())).T
    std = pd.DataFrame(df.apply(np.std)).T
    skew = pd.DataFrame(df.apply(lambda x: x.skew())).T
    kurtosis = pd.DataFrame(df.apply(lambda x: x.kurtosis())).T

    # Metrics Concatenation
    m = pd.concat([min_, max_, range_, mean, median, std, skew, kurtosis]).T.reset_index()
    m.columns = ['attributes', 'min', 'max', 'range', 'mean', 'median', 'std', 'skew', 'kurtosis']
    
    return m