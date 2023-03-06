import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import layers
from copy import deepcopy


class LSTM:
    def __init__(self, look_back=60):
        self.look_back = look_back
        self.model = None
        self.scaler = None
        self.X = None
        self.y = None
        self.history = None
        self.predictions = None
        self.data = pd.DataFrame()


    def get_data(self, ticker, start_date, end_date):
        # get data from yahoo finance
        self.raw_data = yf.download(ticker, start_date, end_date)
        self.data['Target'] = np.sign(self.raw_data['Close'] - self.raw_data['Close'].shift(-1))
        self.data = self.data[['Target']]
        self.data = self.data.dropna()
        self.dates = self.data.index


    def split_data(self, split=[0.8, 0.9]):
        # split data into train, validation, and test sets
        X, y = self.create_dataset()
        s1 = int(len(self.dates)*split[0])
        s2 = int(len(self.dates)*split[1])
        self.dates = {'train': self.dates[:s1], 'val': self.dates[s1:s2], 'test': self.dates[s2:]}
        self.X = {'train': X[:s1], 'val': X[s1:s2], 'test': X[s2:]}
        self.y = {'train': y[:s1], 'val': y[s1:s2], 'test': y[s2:]}


    # def scale_data(self):
    #     self.scaler = MinMaxScaler()
    #     self.scaler.fit(self.train)
    #     self.train = self.scaler.transform(self.train)
    #     self.test = self.scaler.transform(self.test)
    #     self.train_X, self.train_y = self.create_dataset(self.train)
    #     self.test_X, self.test_y = self.create_dataset(self.test)


    def create_dataset(self):
        # create a dataset for training
        X, y = [], []
        dataset = self.data.to_numpy()
        for i in range(len(dataset) - self.look_back - 1):
            a = dataset[i:(i + self.look_back), :]
            X.append(a)
            y.append(dataset[i + self.look_back, :])
        self.dates = self.dates[self.look_back + 1:]
        return np.array(X), np.array(y)
    

    def build_model(self):
        # build a lstm model for algorithmic trading
        self.model = Sequential([layers.Input((self.look_back, 1)),
                    layers.LSTM(256),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])

        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(), 
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])
    

    def train_model(self, epochs=100):
        self.history = self.model.fit(self.X['train'], 
                                      self.y['train'], 
                                      epochs=epochs,
                                      verbose=0,
                                      validation_data=(self.X['val'], self.y['val']))
    

    def predict(self):
        self.pred = {'train': self.model.predict(self.X['train']).flatten(),
                    'val': self.model.predict(self.X['val']).flatten(),
                    'test': self.model.predict(self.X['test']).flatten()}
    

    def rec_predict(self):
        self.rec_pred = []
        self.rec_dates = np.concatenate([self.dates['val'], self.dates['test']])
        last_window = deepcopy(self.X['train'][-1])

        for _ in self.rec_dates:
            next_pred = self.model.predict(np.array([last_window])).flatten()
            self.rec_pred.append(next_pred)
            np.roll(last_window, -1, axis=0)
            last_window[-1] = next_pred
            


    def plot_stock(self):
        plt.figure(figsize=(20, 10))
        plt.plot(self.raw_data.index, self.raw_data['Close'])
        plt.title('Stock Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(['Target'])
        plt.show()


    def plot_split(self):
        for split in ['train', 'val', 'test']:
            plt.plot(self.dates[split], self.y[split])
        plt.title('Stock Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(['train', 'val', 'test'])
        plt.show()


    def plot_history(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'])
        plt.show()


    def plot_predictions(self, separate=False):
        if separate:
            fig, ax = plt.subplots(ncols=3, figsize=(30,10))
            for i, split in enumerate(['train', 'val', 'test']):
                ax[i].plot(self.dates[split], self.pred[split])
                ax[i].plot(self.dates[split], self.y[split])
                ax[i].legend([f'{split} Pred', f'{split} Real'])
        else:
            plt.figure(figsize=(20, 10))
            for split in ['train', 'val', 'test']:
                plt.plot(self.dates[split], self.pred[split])
            plt.plot(self.data.index, self.data['Target'])
            plt.title('Stock Price')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend(['train', 'val', 'test', 'real'])
            
        plt.show()


    def plot_rec_predictions(self):
        plt.figure(figsize=(20, 10))
        plt.plot(self.rec_dates, self.rec_pred)
        plt.plot(self.data.index, self.data['Close'])
        plt.title('Stock Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(['rec', 'real'])
        plt.show()
        