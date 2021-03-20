# import py-torch lightning and py-torch forecasting
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

import pandas as pd

target_cols = ['Max Speed', 'Average Speed']

data = pd.read_csv('Data/nick_running.csv')
data = data.set_index('Activity Date')[target_cols].reset_index()
data['Activity Date'] = pd.to_datetime(data['Activity Date'], format='%Y-%m-%D')
data['all'] = 'true'

data["Activity Date"].dt.monthdata["time_idx"] -= data["time_idx"].min()
data["time_idx"] = data["Activity Date"].dt.year * 12 + data["Activity Date"].dt.monthdata["time_idx"] 


# define dataset
max_encode_length = 36
max_prediction_length = 6
training_cutoff = "2021-02-01"  # day for cutoff

training = TimeSeriesDataSet(
    data[lambda x: x['Activity Date'] < training_cutoff],
    time_idx="Activity Date",
    target='Average Speed',
    # weight="weight",
    group_ids=['all'],
    max_encoder_length=max_encode_length,
    max_prediction_length=max_prediction_length,
    #static_categoricals=[ ... ],
    #static_reals=[ ... ],
    #time_varying_known_categoricals=[ ... ],
    #time_varying_known_reals=[ ... ],
    #time_varying_unknown_categoricals=[ ... ],
    time_varying_unknown_reals=target_cols,
)

