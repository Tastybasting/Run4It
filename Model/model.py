from numpy.random import seed
from tensorflow.random import set_seed
seed(42)
set_seed(42)

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

from multiprocessing import Pool, cpu_count

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM

import warnings
warnings.filterwarnings("ignore")



class ForecastPreprocessor:
    """A pandas dataframe is inputted, which gets transformed by an NMF and gets split into train and test sets
    Input: pandas DataFrame
    Output: pre-processor object, that allows user to create a dataset for training an LSTM model"""
    def __init__(self, df : pd.DataFrame, scaler=MinMaxScaler(), n_components=100, max_selected_components=10):
        
        assert df.index.dtype == pd.to_datetime(['2013']).dtype, "Please ensure your DataFrame has a DateTime index"
        assert max_selected_components < n_components, "max_selected components must be less than total n_components for PCA"
        
        self.df = self.remove_nan(df)

        # define PCA and get the best components 
        self.pca = make_pipeline(scaler, PCA(n_components=n_components))
        self.n_components = n_components
        self.n_best_components = self.get_n_best_components(max_selected_components)
        
        # define NMF and apply
        self.nmf = make_pipeline(MinMaxScaler(), NMF(n_components=self.n_best_components))
        self.preprocess()
        
        
    def remove_nan(self, df):
        df = df.interpolate(method='linear', axis=1)  
        return df.fillna(0)
    
    def get_n_best_components(self, max_selected_components):
        """Provides number of PCA features that capture the majority of the variance"""
        self.pca.fit(self.df)
        ratios = self.pca['pca'].explained_variance_ratio_
        
        # check if next ratio value is smaller than 0.05, if it is, it means that we are covering majority of the variance with existing features
        for idx, r in enumerate(ratios[:-1]):
            if ratios[idx+1] < 0.05:
                return idx+1            
            # cap at 10 components max
            if idx > max_selected_components:
                return max_selected_components
            
    def preprocess(self):
        """Transforms data using the nmf model and the number of best components chosen by PCA"""
        nmf_features = self.nmf.fit_transform(self.df)
        self.df = pd.DataFrame(nmf_features, index=self.df.index)
        
    def input_to_3d(self, data):
        """Transforms data in the correct shape for processing with LSTM"""
        return np.reshape(data, (data.shape[0], data.shape[1], self.n_best_components))

    # build X input and Y target variables
    def create_dataset(self, dataset, look_back=1):
        """Assign input and target variables using dataset"""
        X, Y = [], []
        for i in range(look_back, len(dataset)):
            a = dataset[i-look_back:i, :]
            X.append(a)
            Y.append(dataset[i, :])
        return self.input_to_3d(np.array(X)), np.array(Y)
        
    def train_test_split(self, start_date, end_date, lookback=1):
        """Split dataset into training and testing sets"""
        train, test = self.df.loc[start_date:end_date], self.df.loc[end_date:]
        x_train, y_train = self.create_dataset(train.values, lookback)
        x_test, y_test = self.create_dataset(test.values, lookback)
        return x_train, y_train, x_test, y_test
        
class ForecastModel(Model):
    """Given an input/output shape, an LSTM model is built. The user can specify the number of layers and number of units. Built on top of keras Model class.
    Input: Input, output shapes, train and test data
    Output: Object allowing user to train, test and cross-validate an LSTM model"""
    # Initialise variables
    def __init__(self, input_shape, lstm_units=[128, 64], dense_units=[25], output_shape=2):
        super(ForecastModel, self).__init__()
        self.lstm_units = lstm_units
        self.dense_units = dense_units

        self.model = self.build(input_shape, output_shape)
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    # Build model
    def build(self, input_shape, output_shape):
        """Build LSTM model with user defined number of units and layers"""
        model = Sequential()
        
        if len(self.lstm_units) > 1:
            model.add(LSTM(self.lstm_units[0], return_sequences=True, input_shape=input_shape))
            
            for idx, unit in enumerate(self.lstm_units[1:]):
                if idx < (len(self.lstm_units)-2):
                    model.add(LSTM(unit, return_sequences=True))
                else:
                    model.add(LSTM(unit, return_sequences=False))
        
        else:
            model.add(LSTM(self.lstm_units[0], return_sequences=False, input_shape=input_shape))
        
        for unit in self.dense_units:
            model.add(Dense(unit))
        model.add(Dense(output_shape))
        return model
            
    def cross_validate(self, X, y, scaler=None, n_splits=3, epochs=5, batch_size=1, verbose=0):
        """Cross validate model using n_fold cross validation and RMSE scores. Scaled to original form."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        all_scores = []
        for train_index, val_index in tscv.split(X):
            x_train, x_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), verbose=verbose)
            y_pred = self.model.predict(x_val)
            
            if scaler: 
                score = np.sqrt(mean_squared_error(scaler.inverse_transform(y_val), scaler.inverse_transform(y_pred)))
            else:
                score = np.sqrt(mean_squared_error(y_val, y_pred))
            all_scores.append(score)
         
        print(f'Min rmse across folds: {min(all_scores)}')
        return min(all_scores)

